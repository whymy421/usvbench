"""Bounded (tanh-transformed Gaussian) PPO policies for the journal tasks (diagnostic branch yutong/ppo-fair-fix,
2026-11-12).

Why
---
The registered PPO policy is skrl 1.4.3's GaussianMixin with clip_actions=True: an UNBOUNDED Normal whose sample is
clamped to the action box while the log-probability stays that of the unclamped Normal, evaluated at the clamped
value. Once the mean leaves [-1, 1], every clamped sample has an astronomically small density that swings wildly with
tiny parameter changes, so the PPO ratio and skrl's KL estimate blow up. tasks/_shared/squashed_gaussian.py documents
the same defect for SAC (log_prob minima of -9.8e15) and fixes it there; PPO was left on the clipped Gaussian.
Measured on this branch (PROTOCOL_R2.md, 2026-11-12 22:33): in the P1 fortress runs KLAdaptiveLR drove the learning
rate to its 1e-6 floor for the last 30-40 % of the budget, and the ring PPO-GRU seed 90001 diverged (non-finite
policy loss after the rate reached the floor).

What changes (ONLY the action distribution)
-------------------------------------------
    u ~ Normal(mu, sigma),  a = bias + scale * tanh(u)      (bounded by construction, no clamp)
    log p(a) = log Normal(atanh((a - bias) / scale); mu, sigma) - sum_i [log scale_i + log(1 - tanh(u_i)^2)]
    entropy  = one-sample reparameterised estimate of the transformed entropy (pre-squash entropy + log|da/du|);
               the pre-squash entropy alone cannot see a saturated policy (memory: entropy-regulariser-wrong-variable)
    the stored and the replayed log-probability are both computed from the bounded action itself (atanh of the
    clamped-to-open-interval action), so collection and PPO replay describe exactly the same command.
Unchanged: network shapes and parameter names (MLP: the yaml's shared 128-128-64 ELU trunk `net_container`,
`policy_layer`, `value_layer`, `log_std_parameter`; GRU: as scripts/ppo_memory_models.py), the yaml's initial log-std
and log-std clip range, every agent / trainer / G44-G46 setting. The yaml's clip_actions is dropped: tanh bounds the
action and clamping a sample is the defect.

Source: TanhGaussianMixin / SharedTanhActorCritic of the E8 fix (NavRL isaaclab_tasks/direct/my_first_task_e10/agents/
squashed_gaussian.py, 2026), copied; changes: the log-std defaults follow the journal yaml, GRU policy added.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl.models.torch import DeterministicMixin, Model
from skrl.utils.runner.torch import Runner
from torch.distributions import Normal

INVERSE_EPS = 1.0e-6


class TanhGaussianMixin:
    """Gaussian policy transformed exactly into a finite Box action space."""

    def __init__(self, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0, reduction="sum",
                 inverse_eps=INVERSE_EPS, role=""):
        del role
        if not isinstance(self.action_space, gym.spaces.Box):
            raise TypeError("TanhGaussianMixin requires a gymnasium.spaces.Box action space")
        low = torch.as_tensor(self.action_space.low, device=self.device, dtype=torch.float32)
        high = torch.as_tensor(self.action_space.high, device=self.device, dtype=torch.float32)
        if not torch.isfinite(low).all() or not torch.isfinite(high).all():
            raise ValueError("TanhGaussianMixin requires finite action bounds")
        if not torch.all(high > low):
            raise ValueError("every action upper bound must exceed its lower bound")
        self.register_buffer("_tg_scale", 0.5 * (high - low), persistent=False)
        self.register_buffer("_tg_bias", 0.5 * (high + low), persistent=False)
        self._tg_clip_log_std = bool(clip_log_std)
        self._tg_log_std_min = float(min_log_std)
        self._tg_log_std_max = float(max_log_std)
        self._tg_inverse_eps = float(inverse_eps)
        self._tg_log_std = None
        self._tg_num_samples = None
        self._tg_distribution = None
        reductions = {"mean": torch.mean, "sum": torch.sum, "prod": torch.prod, "none": None}
        if reduction not in reductions:
            raise ValueError("reduction must be one of: mean, sum, prod, none")
        self._tg_reduction = reductions[reduction]

    @staticmethod
    def _unit_tanh_log_det(pre_tanh):
        """Stable log(1 - tanh(x)^2)."""
        return 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))

    def _pre_tanh_and_log_det(self, actions):
        unit = ((actions - self._tg_bias) / self._tg_scale).clamp(-1.0 + self._tg_inverse_eps,
                                                                  1.0 - self._tg_inverse_eps)
        pre_tanh = torch.atanh(unit)
        return pre_tanh, self._tg_scale.log() + self._unit_tanh_log_det(pre_tanh)

    def act(self, inputs: Mapping[str, Any], role: str = ""):
        mean, log_std, outputs = self.compute(inputs, role)
        if self._tg_clip_log_std:
            log_std = log_std.clamp(self._tg_log_std_min, self._tg_log_std_max)
        self._tg_log_std = log_std
        self._tg_num_samples = mean.shape[0]
        self._tg_distribution = Normal(mean, log_std.exp())
        actions = inputs.get("taken_actions")
        if actions is None:
            actions = self._tg_bias + self._tg_scale * torch.tanh(self._tg_distribution.rsample())
        # the latent is reconstructed from the bounded action for collection AND replay (same command, same density)
        pre_tanh, log_det = self._pre_tanh_and_log_det(actions)
        log_prob = self._tg_distribution.log_prob(pre_tanh) - log_det
        if self._tg_reduction is not None:
            log_prob = self._tg_reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)
        outputs["mean_actions"] = self._tg_bias + self._tg_scale * torch.tanh(mean)
        outputs["latent_mean_actions"] = mean
        return actions, log_prob, outputs

    def get_entropy(self, role: str = ""):
        del role
        if self._tg_distribution is None:
            return torch.tensor(0.0, device=self.device)
        pre_tanh = self._tg_distribution.rsample()
        log_det = self._tg_scale.log() + self._unit_tanh_log_det(pre_tanh)
        return self._tg_distribution.entropy() + log_det        # one-sample estimate of the transformed entropy

    def get_log_std(self, role: str = ""):
        del role
        return self._tg_log_std.repeat(self._tg_num_samples, 1)

    def distribution(self, role: str = ""):
        del role
        return self._tg_distribution            # skrl PPO logs .stddev only: the latent exploration scale


class SharedTanhActorCritic(TanhGaussianMixin, DeterministicMixin, Model):
    """The yaml's shared ELU trunk with a bounded actor head and a scalar critic head (same parameter names)."""

    def __init__(self, observation_space, action_space, device, hidden_layers=(128, 128, 64), initial_log_std=-1.9,
                 clip_log_std=True, min_log_std=-20.0, max_log_std=2.0):
        Model.__init__(self, observation_space, action_space, device)
        TanhGaussianMixin.__init__(self, clip_log_std=clip_log_std, min_log_std=min_log_std, max_log_std=max_log_std)
        DeterministicMixin.__init__(self, clip_actions=False)
        layers, size = [], self.num_observations
        for width in hidden_layers:
            layers += [nn.Linear(size, int(width)), nn.ELU()]
            size = int(width)
        self.net_container = nn.Sequential(*layers)
        self.policy_layer = nn.Linear(size, self.num_actions)
        self.value_layer = nn.Linear(size, 1)
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), float(initial_log_std)))
        self._shared_output = None
        self.to(device)

    def act(self, inputs, role=""):
        if role == "policy":
            return TanhGaussianMixin.act(self, inputs, role)
        if role == "value":
            return DeterministicMixin.act(self, inputs, role)
        raise ValueError("unknown model role: %r" % role)

    def compute(self, inputs, role=""):
        if role == "policy":
            shared = self.net_container(inputs["states"])
            self._shared_output = shared
            return self.policy_layer(shared), self.log_std_parameter, {}
        if role == "value":
            shared = self._shared_output                   # skrl calls policy then value on the same states
            if shared is None:
                shared = self.net_container(inputs["states"])
            self._shared_output = None
            return self.value_layer(shared), {}
        raise ValueError("unknown model role: %r" % role)


def _yaml_policy_settings(cfg):
    p = cfg["models"]["policy"]
    block = (p.get("network") or [{}])[0]
    if str(block.get("activations", "elu")).lower() != "elu":
        raise ValueError("SharedTanhActorCritic mirrors an ELU trunk; the yaml says %r" % block.get("activations"))
    return dict(hidden_layers=tuple(int(w) for w in block.get("layers", (128, 128, 64))),
                initial_log_std=float(p["initial_log_std"]), clip_log_std=bool(p.get("clip_log_std", True)),
                min_log_std=float(p.get("min_log_std", -20.0)), max_log_std=float(p.get("max_log_std", 2.0)))


class SquashedRunner(Runner):
    """skrl Runner with the shared tanh-Gaussian actor-critic in place of the yaml's shared Gaussian model."""

    def _generate_models(self, env, cfg):
        if cfg["models"].get("separate", False):
            raise ValueError("the journal yaml uses a shared policy/value model; SquashedRunner mirrors that")
        model = SharedTanhActorCritic(env.observation_space, env.action_space, env.device,
                                      **_yaml_policy_settings(cfg))
        model.init_state_dict("policy")
        model.init_state_dict("value")
        print("==================================================\nShared tanh-Gaussian actor-critic (bounded "
              "actions)\n==================================================\n%s\n"
              "--------------------------------------------------" % model, flush=True)
        return {"agent": {"policy": model, "value": model}}


def describe(runner):
    model = runner.agent.models["policy"]
    return {"variant": "mlp_tanh", "action_distribution": "tanh-transformed Gaussian (exact log-prob)",
            "parameters": int(sum(p.numel() for p in model.parameters())), "model": str(model)}
