"""Demonstration warm start for the journal PPO (diagnostic branch yutong/ppo-fair-fix, round 6, 2026-11-13).

Why: five rounds of PPO fixes (reward, exploration scale, GRU memory, bounded actions, timeout penalty, gap-width
curriculum) all scored 0/128 on ring and fortress, and PPO almost never reached a goal during training: the blocker
is exploration. A warm start from demonstrations skips it: the classical memory planner (the comparator, 111/128
fortress, 128/128 ring) drives the TRAINING env, the PPO network imitates it, and PPO continues from there.
This is no longer pure PPO (it uses the comparator's trajectories); PROTOCOL_R6.md states what it can answer.

Steps (run inside train_journal_ppo.run_training, after the agent and the G44/G46 hooks exist, before the trainer):
  collect   the classical planner (planner_los_pid_mem_v2, the settings of scripts/eval_classical_calm.py) acts in the
            run's own env (training seed, disturbance mixture, tier, fair-fix reward) for `steps` control steps; per
            step the PPO observation (the skrl wrapper's), the planner action, the reward incl. the G44/G46 failure
            penalty (a separate FailurePenalty, so the run's ppo_fix.json counts PPO only) and the done flag.
            The steps go through the wrapper's ORIGINAL step (not the StepMonitor), so the PPO budget, the tracker
            and the normaliser-freeze clock are untouched; the heartbeat is written every 32 steps.
  fit       the state / value RunningStandardScalers take the demonstration statistics (count = samples; they keep
            updating afterwards and freeze at 5 % as always); the policy mean tanh(mu) regresses the planner action
            (MSE), the value head the discounted demonstration return (gamma of the agent). Samples of an episode
            unfinished at the end of the collection with < TAIL_STEPS steps left are excluded from the value loss.
            log_std stays at its yaml initial value. MLP (shared trunk): minibatches. GRU: the 64 time series in
            time order, hidden state carried across windows of WINDOW steps (detached) and zeroed after every done,
            exactly as while acting.
  save      the imitation policy (before PPO) as bc_init/agent_0.pt, the g46_init whole-agent format, for evaluation.
PPO then continues from the env state at the end of the collection (no reset; the skrl wrapper returns its cached
observation).
"""
from __future__ import annotations

import importlib
import math
import time
from pathlib import Path

BC_CHECKPOINT = "bc_init/agent_0.pt"
TAIL_STEPS = 3000
WINDOW = 64
DEFAULTS = {"steps": 6400, "epochs_mlp": 30, "epochs_gru": 20, "batch": 4096, "lr": 1.0e-3, "value_weight": 0.5,
            "tail_steps": TAIL_STEPS, "window": WINDOW, "expert": "planner_los_pid_mem_v2"}
# classical_planner_pid CLI defaults (as scripts/eval_classical_calm.py)
REPLAN_EVERY, WAYPOINT_SPACING_REF, ACCEPT_RADIUS_REF, PLAN_CELL_REF = 60, 2.0, 1.0, 0.25


def make_expert(base, env_cfg, package_root):
    import classical_planner_pid as C
    import journal_controllers as jc
    for family, geometry in (("hazard_nav", "hazard_geometry"), ("path_hazard", "path_hazard_geometry")):
        try:                                     # pin the planner to the env's geometry (eval_classical_calm note)
            C._GEOMETRY_CACHE[family] = importlib.import_module("%s.%s.%s" % (package_root, family, geometry))
        except ImportError:
            pass
    vessel = str(env_cfg.vehicle)
    scale = jc.length_scale(vessel)
    settings = dict(replan_every=REPLAN_EVERY, waypoint_spacing_m=WAYPOINT_SPACING_REF * scale,
                    accept_radius_m=ACCEPT_RADIUS_REF * scale, plan_cell_m=PLAN_CELL_REF * scale,
                    gains=jc.load_gains(vessel), max_range_m=jc.ray_max_range(env_cfg))
    return C.JournalMemoryPlannerController(base, ladder=C.journal_memory_ladder(vessel), vessel=vessel, **settings)


def collect(step_fn, reset_fn, base, expert, penalty, codes_fn, steps, code9_reward, heartbeat=None, log=print):
    import torch
    import torch.nn.functional as F  # noqa: F401
    obs, _ = reset_fn()
    n, f = obs.shape
    penalty.refresh()
    obs_buf = torch.empty((steps, n, f), dtype=torch.float32)
    act_buf = None
    rew_buf = torch.zeros((steps, n), dtype=torch.float32)
    done_buf = torch.zeros((steps, n), dtype=torch.bool)
    episodes = successes = 0
    outcomes = {}
    t0 = time.monotonic()
    for t in range(steps):
        with torch.inference_mode():
            action = expert.act()
        action = action.detach().clone().float()
        if act_buf is None:
            act_buf = torch.empty((steps, n, action.shape[-1]), dtype=torch.float32)
        obs_buf[t] = obs.detach().float().cpu()
        act_buf[t] = action.cpu()
        out = step_fn(action.to(obs.device))
        obs, rew, term, trunc = out[0], out[1], out[2], out[3]
        rew = torch.nan_to_num(rew.detach().float().reshape(-1).cpu(), nan=code9_reward, posinf=code9_reward,
                               neginf=code9_reward)
        done = (term.reshape(-1) | trunc.reshape(-1)).detach().cpu().bool()
        dones = done.tolist()
        codes = codes_fn()
        for i, value in penalty.penalties(dones, codes).items():
            rew[i] += value
        if any(dones):
            rows = [i for i, d in enumerate(dones) if d]
            penalty.refresh(rows)
            for i in rows:
                episodes += 1
                successes += int(bool(base.episode_success[i]))
                name = str(int(codes[i])) if codes is not None else "?"
                outcomes[name] = outcomes.get(name, 0) + 1
        expert.reset(done)
        rew_buf[t] = rew
        done_buf[t] = done
        if heartbeat is not None and (t + 1) % 32 == 0:
            heartbeat()
        if (t + 1) % 640 == 0:
            log("[demo] step %d/%d, episodes %d, successes %d, %.1f steps/s"
                % (t + 1, steps, episodes, successes, (t + 1) / max(1e-9, time.monotonic() - t0)))
    stats = {"steps": steps, "num_envs": n, "samples": steps * n, "episodes_finished": episodes,
             "successes": successes, "success_rate": (successes / episodes) if episodes else None,
             "termination_codes": dict(sorted(outcomes.items())), "wall_s": round(time.monotonic() - t0, 1),
             "penalty": penalty.summary()}
    return {"obs": obs_buf, "act": act_buf, "rew": rew_buf, "done": done_buf}, stats


def returns(rew, done, gamma, tail_steps):
    import torch
    import torch.nn.functional as F  # noqa: F401
    """Discounted return per sample (time-limit ends are terminal: time_limit_bootstrap is False) and the value mask."""
    steps, n = rew.shape
    g = torch.zeros(n)
    out = torch.zeros_like(rew)
    for t in range(steps - 1, -1, -1):
        g = rew[t] + gamma * (~done[t]).float() * g
        out[t] = g
    any_done = done.any(dim=0)
    last_done = torch.where(any_done, steps - 1 - done.flip(0).float().argmax(dim=0), torch.full((n,), -1))
    t_index = torch.arange(steps).unsqueeze(1)
    unfinished = t_index > last_done.unsqueeze(0)
    valid = ~unfinished | ((steps - t_index) >= tail_steps)
    return out, valid


def _set_scaler(scaler, x):
    import torch
    import torch.nn.functional as F  # noqa: F401
    x = x.double()
    with torch.no_grad():
        scaler.running_mean = x.mean(dim=0).to(scaler.running_mean.device)
        scaler.running_variance = x.var(dim=0).to(scaler.running_variance.device)
        scaler.current_count = torch.tensor(float(x.shape[0]), dtype=torch.float64,
                                            device=scaler.current_count.device)


def _bounded(model, latent):
    import torch
    import torch.nn.functional as F  # noqa: F401
    return model._tg_bias + model._tg_scale * torch.tanh(latent)


def fit(agent, data, gamma, cfg, recurrent, log=print):
    import torch
    import torch.nn.functional as F  # noqa: F401
    obs, act, rew, done = data["obs"], data["act"], data["rew"], data["done"]
    steps, n, f = obs.shape
    ret, valid = returns(rew, done, gamma, int(cfg["tail_steps"]))
    sp, vp = agent._state_preprocessor, agent._value_preprocessor
    _set_scaler(sp, obs.reshape(-1, f))
    _set_scaler(vp, ret[valid].reshape(-1, 1))
    with torch.no_grad():
        x = sp(obs.reshape(-1, f)).reshape(steps, n, f)
        y = vp(ret.reshape(-1, 1)).reshape(steps, n)
    policy, value = agent.models["policy"], agent.models["value"]
    params = list({id(p): p for p in list(policy.parameters()) + list(value.parameters())}.values())
    params = [p for p in params if p is not getattr(policy, "log_std_parameter", None)]
    opt = torch.optim.Adam(params, lr=float(cfg["lr"]))
    w = float(cfg["value_weight"])
    history = []
    t0 = time.monotonic()
    if not recurrent:
        X, A, Y, M = x.reshape(-1, f), act.reshape(-1, act.shape[-1]), y.reshape(-1), valid.reshape(-1)
        for epoch in range(int(cfg["epochs_mlp"])):
            perm = torch.randperm(X.shape[0])
            sums = [0.0, 0.0, 0]
            for k in range(0, X.shape[0], int(cfg["batch"])):
                idx = perm[k:k + int(cfg["batch"])]
                h = policy.net_container(X[idx])
                loss_a = F.mse_loss(_bounded(policy, policy.policy_layer(h)), A[idx])
                m = M[idx]
                loss_v = F.mse_loss(policy.value_layer(h).squeeze(-1)[m], Y[idx][m]) if bool(m.any()) else h.sum() * 0
                opt.zero_grad()
                (loss_a + w * loss_v).backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sums[0] += float(loss_a) * len(idx)
                sums[1] += float(loss_v) * len(idx)
                sums[2] += len(idx)
            history.append({"epoch": epoch + 1, "action_mse": sums[0] / sums[2], "value_mse": sums[1] / sums[2]})
            log("[bc] epoch %d action mse %.5f value mse %.4f (%.0f s)"
                % (epoch + 1, history[-1]["action_mse"], history[-1]["value_mse"], time.monotonic() - t0))
    else:
        window = int(cfg["window"])
        for epoch in range(int(cfg["epochs_gru"])):
            hp = torch.zeros(1, n, policy.hidden_size)
            hv = torch.zeros(1, n, value.hidden_size)
            sums = [0.0, 0.0, 0]
            for t0w in range(0, steps, window):
                t1w = min(steps, t0w + window)
                outs_p, outs_v = [], []
                for t in range(t0w, t1w):
                    op, hp = policy.gru(x[t].unsqueeze(1), hp)
                    ov, hv = value.gru(x[t].unsqueeze(1), hv)
                    outs_p.append(op[:, 0])
                    outs_v.append(ov[:, 0])
                    keep = (~done[t]).float().view(1, n, 1)
                    hp, hv = hp * keep, hv * keep
                P = torch.stack(outs_p)                                   # (L, n, H)
                V = torch.stack(outs_v)
                loss_a = F.mse_loss(_bounded(policy, policy.head(P)), act[t0w:t1w])
                m = valid[t0w:t1w]
                loss_v = (F.mse_loss(value.head(V).squeeze(-1)[m], y[t0w:t1w][m]) if bool(m.any())
                          else V.sum() * 0)
                opt.zero_grad()
                (loss_a + w * loss_v).backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                hp, hv = hp.detach(), hv.detach()
                k = (t1w - t0w) * n
                sums[0] += float(loss_a) * k
                sums[1] += float(loss_v) * k
                sums[2] += k
            history.append({"epoch": epoch + 1, "action_mse": sums[0] / sums[2], "value_mse": sums[1] / sums[2]})
            log("[bc] epoch %d action mse %.5f value mse %.4f (%.0f s)"
                % (epoch + 1, history[-1]["action_mse"], history[-1]["value_mse"], time.monotonic() - t0))
    for p in params:
        if p.grad is not None:
            p.grad = None
    action_var = float(act.reshape(-1, act.shape[-1]).var(dim=0).mean())
    return {"history": history, "final_action_mse": history[-1]["action_mse"],
            "action_variance": action_var, "value_samples": int(valid.sum()),
            "return_mean": float(ret[valid].mean()), "return_std": float(ret[valid].std()),
            "fit_wall_s": round(time.monotonic() - t0, 1)}


def save_agent(agent, path, torch_module=None):
    if torch_module is None:
        import torch as torch_module
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=False)
    modules = {name: (m.state_dict() if hasattr(m, "state_dict") else m) for name, m in agent.checkpoint_modules.items()}
    tmp = path.with_name(path.name + ".tmp")
    torch_module.save(modules, str(tmp))
    import os
    os.replace(str(tmp), str(path))           # atomic: the bc evaluation starts as soon as the file exists
    return str(path)


def warm_start(*, agent, wrapped, step_fn, base, env_cfg, package_root, failure, carrier, num_envs, codes_fn,
               code9_reward, gamma, recurrent, out, cfg, heartbeat=None, log=print):
    from ppo_training_fixes import FailurePenalty
    expert = make_expert(base, env_cfg, package_root)
    penalty = FailurePenalty(failure.terms, base, carrier, num_envs, codes=failure.codes)
    data, demo = collect(step_fn, wrapped.reset, base, expert, penalty, codes_fn, int(cfg["steps"]),
                         float(code9_reward), heartbeat=heartbeat, log=log)
    log("[demo] done: %s" % {k: v for k, v in demo.items() if k != "penalty"})
    fitted = fit(agent, data, float(gamma), cfg, recurrent, log=log)
    path = save_agent(agent, Path(out) / BC_CHECKPOINT)
    if recurrent and hasattr(agent, "_rnn_initial_states"):          # PPO starts with zero hidden states
        for role in agent._rnn_initial_states:
            for s in agent._rnn_initial_states[role]:
                s.zero_()
    info = {"config": dict(cfg), "demonstrations": demo, "fit": fitted, "checkpoint": BC_CHECKPOINT,
            "checkpoint_path": path}
    if cfg.get("anchor"):                                             # round 7: kept for the anchor and resumes
        save_data(data, Path(out) / DEMO_DATA)
        info["demo_data"] = DEMO_DATA
    return info, data


# ------------------------------------------------------------------ round 7: demonstration anchor
DEMO_DATA = "demo_data.pt"
ANCHOR_DEFAULTS = {"freeze_normaliser_at_start": True, "lr": 1.0e-4, "schedule": "linear 1 -> 0 over the budget",
                   "steps_per_update_mlp": 8, "batch_mlp": 4096, "steps_per_update_gru": 2,
                   "window_gru": 128, "burn_in_gru": 64}


def save_data(data, path):
    import os
    import torch
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({k: v for k, v in data.items()}, str(tmp))
    os.replace(str(tmp), str(path))


def load_data(path):
    import torch
    return torch.load(str(path), map_location="cpu", weights_only=False)


def freeze_now(freeze, timestep):
    """Freeze the observation normaliser at once (round 7): the demonstration statistics stay fixed."""
    import ppo_training_fixes as FX
    if not freeze.frozen:
        freeze.frozen = True
        freeze.frozen_at = int(timestep)
        freeze.digest_at_freeze = FX.module_digest(freeze.scaler) if freeze.scaler is not None else None
    return freeze.frozen_at


class BCAnchor:
    """After every PPO update: a few imitation steps on the demonstrations (policy only, log_std untouched), with
    their own Adam at lr * w(t), w(t) = 1 - t / budget. Requires the frozen observation normaliser (the normalised
    demonstrations are computed once)."""

    def __init__(self, agent, data, cfg, budget, timestep_fn, recurrent):
        import torch
        self.agent, self.cfg, self.budget, self.timestep_fn, self.recurrent = agent, dict(cfg), int(budget),             timestep_fn, bool(recurrent)
        obs = data["obs"]
        steps, n, f = obs.shape
        with torch.no_grad():
            self.x = agent._state_preprocessor(obs.reshape(-1, f)).reshape(steps, n, f).detach()
        self.act, self.done = data["act"], data["done"]
        self.policy = agent.models["policy"]
        log_std = getattr(self.policy, "log_std_parameter", None)
        self.params = [p for p in self.policy.parameters() if p is not log_std]
        self.opt = torch.optim.Adam(self.params, lr=float(self.cfg["lr"]))
        self.updates = 0
        self.steps = 0
        self.history = []

    def weight(self):
        return max(0.0, 1.0 - float(self.timestep_fn()) / self.budget)

    def _loss(self):
        import torch
        import torch.nn.functional as F
        steps, n, f = self.x.shape
        p = self.policy
        if not self.recurrent:
            idx = torch.randint(0, steps * n, (int(self.cfg["batch_mlp"]),))
            X = self.x.reshape(-1, f)[idx]
            A = self.act.reshape(-1, self.act.shape[-1])[idx]
            return F.mse_loss(_bounded(p, p.policy_layer(p.net_container(X))), A)
        window, burn = int(self.cfg["window_gru"]), int(self.cfg["burn_in_gru"])
        t0 = int(torch.randint(0, steps - window + 1, (1,)))
        h = torch.zeros(1, n, p.hidden_size)
        outs = []
        for t in range(t0, t0 + window):
            o, h = p.gru(self.x[t].unsqueeze(1), h)
            if t >= t0 + burn:
                outs.append(o[:, 0])
            h = h * (~self.done[t]).float().view(1, n, 1)
        return F.mse_loss(_bounded(p, p.head(torch.stack(outs))), self.act[t0 + burn:t0 + window])

    def after_update(self):
        import torch
        self.updates += 1
        w = self.weight()
        if w <= 0.0:
            return
        for group in self.opt.param_groups:
            group["lr"] = float(self.cfg["lr"]) * w
        k = int(self.cfg["steps_per_update_gru" if self.recurrent else "steps_per_update_mlp"])
        total = 0.0
        for _ in range(k):
            loss = self._loss()
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            self.opt.step()
            total += float(loss)
            self.steps += 1
        for q in self.params:
            q.grad = None
        if self.updates % 20 == 1:
            self.history.append({"timestep": int(self.timestep_fn()), "weight": round(w, 4),
                                 "bc_mse": total / k})

    def summary(self):
        return {"config": self.cfg, "updates_seen": self.updates, "bc_steps": self.steps, "history": self.history}


def install_anchor(agent, data, cfg, budget, timestep_fn, recurrent, freeze, out):
    """Freeze the normaliser now and run BCAnchor.after_update after every agent._update (after the monitor's
    divergence check); the anchor's summary is rewritten to demo_anchor.json every 20 updates."""
    import json
    frozen_at = freeze_now(freeze, timestep_fn())
    anchor = BCAnchor(agent, data, cfg, budget, timestep_fn, recurrent)
    update = agent._update
    path = Path(out) / "demo_anchor.json"

    def anchored_update(*args, **kwargs):
        result = update(*args, **kwargs)
        anchor.after_update()
        if anchor.updates % 20 == 1:
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(dict(anchor.summary(), normaliser_frozen_at=frozen_at), indent=1))
            import os
            os.replace(str(tmp), str(path))
        return result

    agent._update = anchored_update
    return anchor
