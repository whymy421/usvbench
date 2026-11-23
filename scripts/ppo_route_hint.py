"""Route hint for the journal PPO (diagnostic branch yutong/ppo-fair-fix, round 10, 2026-11-22).

Why: nine rounds of PPO fixes left fortress and ring at 0/128 (one single-seed 15/128 that did not replicate), while
the classical comparator planner_los_pid_mem_v2 reaches 111/128 and 128/128. The comparator has a ray-return MEMORY
and a shortest-path planner on it; PPO has neither (its observation points straight at the goal). Without the memory
the classical planner itself scores 0/128 on fortress. This module gives PPO the SAME route information the
comparator steers by, so the comparison is information-matched.

What it does (PROTOCOL_R10.md): the comparator's own planner object (ppo_demo_warmstart.make_expert, i.e.
classical_planner_pid.JournalMemoryPlannerController with the eval_classical_calm settings) runs alongside the env.
After the env has assembled its observation (so the D5 estimate of this step exists), the planner reads exactly what
it reads in its own evaluation -- the D5 position estimate, the same 36-ray block, the heading -- updates its memory,
replans when due and advances along its plan. The policy observation's goal-bearing pair (columns 0 and 1: dot and
cross of the bow with the unit vector to the goal) is replaced by the same pair towards the planner's current
waypoint. Every other column is unchanged (distance-to-goal, velocities, rays), so the network shape is unchanged.

* The planner's ACTION is discarded: the policy still chooses thrust and yaw.
* Fallback to the original goal bearing (counted): a held env (the planner found no complete route and holds at the
  end of its truncated plan) and a waypoint closer than 1e-3 m.
* Reset: the env's _reset_idx is wrapped so the done envs' planner state and memory are cleared, as in the
  comparator's own evaluation loop (reset(done) after the step).
* Information: the planner never reads obstacle_centers/radii/active, D7 objects, sea, current or wind (its class
  docstring); it reads the D5 estimate the observation is built from. Same inputs as the PPO observation, plus the
  memory and the planning computation.
The SAME hook is installed by train_journal_ppo.py (--route-hint) and eval_fair_fix_calm.py (--route-hint), so the
policy sees the hint while training and while being evaluated.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

VERSION = "r10_route_hint_v1"
DEFAULTS = {"version": VERSION, "planner": "planner_los_pid_mem_v2 (ppo_demo_warmstart.make_expert settings)",
            "columns_replaced": [0, 1], "columns_meaning": "dot, cross of the bow with the unit vector to the "
            "planner's current waypoint (was: to the goal)", "fallback": "goal bearing when held or < 1e-3 m"}
MIN_DISTANCE_M = 1.0e-3


class RouteHint:
    def __init__(self, base, env_cfg, package_root, torch_module):
        import ppo_demo_warmstart as DW
        self.torch = torch_module
        self.base = base
        self.expert = DW.make_expert(base, env_cfg, package_root)
        self.num_envs = int(base.num_envs)
        self.calls = 0
        self.rows_hinted = 0
        self.rows_fallback_held = 0
        self.rows_fallback_close = 0
        self.resets = 0
        self.last_angle_change = None
        self._orig_obs = base._get_observations
        self._orig_reset = base._reset_idx
        base._get_observations = self._get_observations
        base._reset_idx = self._reset_idx

    # -- env hooks ----------------------------------------------------------
    def _reset_idx(self, env_ids):
        result = self._orig_reset(env_ids)
        t = self.torch
        mask = t.zeros(self.num_envs, dtype=t.bool)
        if env_ids is None:
            mask[:] = True
        else:
            ids = t.as_tensor(env_ids).reshape(-1).to("cpu").long()
            mask[ids] = True
        self.resets += int(mask.sum())
        self.expert.reset(mask)
        return result

    def _get_observations(self):
        out = self._orig_obs()
        t = self.torch
        with t.no_grad():
            obs = out["policy"]
            self.expert.act()                         # memory update, replan when due, advance along the plan
            targets = []
            for i in range(self.num_envs):
                plan = self.expert.plans[i]
                targets.append(plan[int(self.expert.wp_index[i])])
            position = self.expert._inputs.position_world
            import numpy as np
            target = t.as_tensor(np.stack(targets), dtype=position.dtype, device=position.device)
            to_target = target - position
            distance = to_target.norm(dim=-1)
            held = t.as_tensor(self.expert.held, device=position.device)
            close = distance < MIN_DISTANCE_M
            use = ~(held | close)
            direction = to_target / distance.clamp_min(MIN_DISTANCE_M).unsqueeze(-1)
            forward = self.base._forward_2d().to(direction.dtype)
            dot = (forward * direction).sum(dim=-1)
            cross = forward[:, 0] * direction[:, 1] - forward[:, 1] * direction[:, 0]
            new = obs.clone()
            use_d = use.to(obs.device)
            new[:, 0] = t.where(use_d, dot.to(obs.dtype).to(obs.device), obs[:, 0])
            new[:, 1] = t.where(use_d, cross.to(obs.dtype).to(obs.device), obs[:, 1])
            self.calls += 1
            self.rows_hinted += int(use.sum())
            self.rows_fallback_held += int(held.sum())
            self.rows_fallback_close += int((close & ~held).sum())
            if self.calls % 500 == 1:                  # how far the hint turns the bearing (descriptive)
                goal_angle = t.atan2(obs[:, 1], obs[:, 0])
                hint_angle = t.atan2(new[:, 1], new[:, 0])
                diff = t.remainder(hint_angle - goal_angle + 3.141592653589793, 6.283185307179586) - 3.141592653589793
                self.last_angle_change = {"call": self.calls, "median_abs_deg": round(float(diff.abs().median()) * 57.29578, 2),
                                          "max_abs_deg": round(float(diff.abs().max()) * 57.29578, 2)}
        out = dict(out)
        out["policy"] = new
        return out

    # -- evidence -----------------------------------------------------------
    def summary(self):
        e = self.expert
        rows = max(1, self.calls * self.num_envs)
        return dict(DEFAULTS, calls=self.calls, rows=self.calls * self.num_envs, rows_hinted=self.rows_hinted,
                    fraction_hinted=round(self.rows_hinted / rows, 4),
                    rows_fallback_held=self.rows_fallback_held, rows_fallback_close=self.rows_fallback_close,
                    env_resets=self.resets, plan_calls=int(e.plan_calls), unplannable_legs=int(e.unplannable_legs),
                    escape_plans=int(e.escape_plans), escape_failures=int(e.escape_failures),
                    straight_fallbacks=int(e.straight_fallbacks), last_angle_change=self.last_angle_change)

    def write(self, path):
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(self.summary(), indent=1), encoding="utf-8")
            os.replace(str(tmp), str(path))
        except OSError as exc:                         # evidence only: never kill a run for it
            print("ppo_route_hint: WARNING %s not written: %s" % (path, exc), flush=True)


def install(base, env_cfg, package_root, torch_module):
    return RouteHint(base, env_cfg, package_root, torch_module)
