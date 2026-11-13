"""Recurrent PPO (PPO-GRU) for the journal tasks (diagnostic branch yutong/ppo-fair-fix, 2026-11-12).

Why: the classical planner and MPPI keep a per-episode memory of the ray returns (planner_memory.py; the G22
replica took the classical planner from 0/32 to 32/32 on fortress WAM-V) while the registered PPO policy is a
memoryless MLP over the current 42-D observation. PPO-GRU is the standard memory baseline of RL.

What changes against the registration yaml: ONLY the models. Agent, memory, preprocessors, KLAdaptiveLR, every
PPO hyperparameter, the trainer and all train_journal_ppo machinery (G44/G46 fixes, monitor, record) are unchanged.
  policy  GRU(42 -> 128, one layer) -> 128 -> 64 -> actions (ELU), Gaussian head with the yaml's log-std settings
  value   GRU(42 -> 128, one layer) -> 128 -> 64 -> 1 (ELU)
Separate networks (skrl's recurrent PPO keeps one hidden state per model); the yaml's MLP shares a trunk.
skrl's PPO_RNN trains them with truncated back-propagation through time over sequences of `sequence_length`
control steps (default = the yaml's rollouts, 32 steps = 0.53 s at 60 Hz). While acting, the hidden state is
carried through the whole episode and zeroed at every episode end (PPO_RNN.record_transition; the calm evaluator
does the same by hand).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.utils.runner.torch import Runner

POLICY_VARIANTS = ("mlp", "gru")
GRU_HIDDEN = 128
HEAD_LAYERS = (128, 64)


def _head(in_size, out_size):
    layers, size = [], in_size
    for width in HEAD_LAYERS:
        layers += [nn.Linear(size, width), nn.ELU()]
        size = width
    layers.append(nn.Linear(size, out_size))
    return nn.Sequential(*layers)


def _run_gru(gru, states, hidden, terminated, training, sequence_length):
    """skrl's recurrent-model pattern: (N*L, F) sequences while updating, one step per env while acting."""
    if training:
        rnn_input = states.view(-1, sequence_length, states.shape[-1])                  # (N, L, F)
        hidden = hidden.view(gru.num_layers, -1, sequence_length, hidden.shape[-1])     # (layers, N, L, H)
        hidden = hidden[:, :, 0, :].contiguous()                                        # state at sequence start
        if terminated is not None and bool(torch.any(terminated)):
            terminated = terminated.view(-1, sequence_length)
            cuts = ([0] + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
                    + [sequence_length])
            outputs = []
            for i0, i1 in zip(cuts[:-1], cuts[1:]):
                out, hidden = gru(rnn_input[:, i0:i1, :], hidden)
                keep = (~terminated[:, i1 - 1]).to(hidden.dtype).view(1, -1, 1)
                hidden = hidden * keep                                                  # out of place (autograd)
                outputs.append(out)
            rnn_output = torch.cat(outputs, dim=1)
        else:
            rnn_output, hidden = gru(rnn_input, hidden)
    else:
        rnn_output, hidden = gru(states.view(-1, 1, states.shape[-1]), hidden)          # (num_envs, 1, H)
    return torch.flatten(rnn_output, start_dim=0, end_dim=1), hidden


class GRUPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, *, num_envs, sequence_length, initial_log_std,
                 clip_actions=True, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0, hidden_size=GRU_HIDDEN):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std, min_log_std=min_log_std,
                               max_log_std=max_log_std, reduction="sum")
        self.num_envs, self.sequence_length, self.hidden_size = int(num_envs), int(sequence_length), int(hidden_size)
        self.gru = nn.GRU(input_size=self.num_observations, hidden_size=self.hidden_size, num_layers=1,
                          batch_first=True)
        self.head = _head(self.hidden_size, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), float(initial_log_std)))

    def get_specification(self):
        return {"rnn": {"sequence_length": self.sequence_length, "sizes": [(1, self.num_envs, self.hidden_size)]}}

    def compute(self, inputs, role):
        out, hidden = _run_gru(self.gru, inputs["states"], inputs["rnn"][0], inputs.get("terminated"),
                               self.training, self.sequence_length)
        return self.head(out), self.log_std_parameter, {"rnn": [hidden]}


def _gru_tanh_policy_class():
    """GRU policy with the bounded tanh-Gaussian head (scripts/ppo_squashed_models.py); same network as GRUPolicy."""
    from ppo_squashed_models import TanhGaussianMixin

    class GRUTanhPolicy(TanhGaussianMixin, Model):
        def __init__(self, observation_space, action_space, device, *, num_envs, sequence_length, initial_log_std,
                     clip_actions=False, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0,
                     hidden_size=GRU_HIDDEN):
            del clip_actions                 # tanh bounds the action; clamping a sample is the defect being removed
            Model.__init__(self, observation_space, action_space, device)
            TanhGaussianMixin.__init__(self, clip_log_std=clip_log_std, min_log_std=min_log_std,
                                       max_log_std=max_log_std)
            self.num_envs, self.sequence_length, self.hidden_size = int(num_envs), int(sequence_length), int(hidden_size)
            self.gru = nn.GRU(input_size=self.num_observations, hidden_size=self.hidden_size, num_layers=1,
                              batch_first=True)
            self.head = _head(self.hidden_size, self.num_actions)
            self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), float(initial_log_std)))
            self.to(device)

        def get_specification(self):
            return {"rnn": {"sequence_length": self.sequence_length, "sizes": [(1, self.num_envs, self.hidden_size)]}}

        def compute(self, inputs, role):
            out, hidden = _run_gru(self.gru, inputs["states"], inputs["rnn"][0], inputs.get("terminated"),
                                   self.training, self.sequence_length)
            return self.head(out), self.log_std_parameter, {"rnn": [hidden]}

    return GRUTanhPolicy


class GRUValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, *, num_envs, sequence_length, clip_actions=False,
                 hidden_size=GRU_HIDDEN):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.num_envs, self.sequence_length, self.hidden_size = int(num_envs), int(sequence_length), int(hidden_size)
        self.gru = nn.GRU(input_size=self.num_observations, hidden_size=self.hidden_size, num_layers=1,
                          batch_first=True)
        self.head = _head(self.hidden_size, 1)

    def get_specification(self):
        return {"rnn": {"sequence_length": self.sequence_length, "sizes": [(1, self.num_envs, self.hidden_size)]}}

    def compute(self, inputs, role):
        out, hidden = _run_gru(self.gru, inputs["states"], inputs["rnn"][0], inputs.get("terminated"),
                               self.training, self.sequence_length)
        return self.head(out), {"rnn": [hidden]}


class RecurrentRunner(Runner):
    """skrl Runner with the GRU models and PPO_RNN; everything else is built from the cfg exactly as Runner does."""

    def __init__(self, env, cfg, sequence_length=None, squash=False):
        self._sequence_length = int(sequence_length or cfg["agent"]["rollouts"])
        self._squash = bool(squash)          # True = bounded tanh-Gaussian head (scripts/ppo_squashed_models.py)
        if int(cfg["agent"]["rollouts"]) % self._sequence_length:
            raise ValueError("sequence_length must divide the rollouts")
        super().__init__(env, cfg)

    def _component(self, name):
        if name.lower() == "ppo":
            from skrl.agents.torch.ppo import PPO_RNN
            return PPO_RNN
        return super()._component(name)

    def _generate_models(self, env, cfg):
        p = cfg["models"]["policy"]
        v = cfg["models"]["value"]
        common = dict(num_envs=env.num_envs, sequence_length=self._sequence_length)
        policy_class = _gru_tanh_policy_class() if self._squash else GRUPolicy
        policy = policy_class(env.observation_space, env.action_space, env.device, **common,
                           initial_log_std=float(p["initial_log_std"]), clip_actions=bool(p.get("clip_actions", True)),
                           clip_log_std=bool(p.get("clip_log_std", True)),
                           min_log_std=float(p.get("min_log_std", -20.0)), max_log_std=float(p.get("max_log_std", 2.0)))
        value = GRUValue(env.observation_space, env.action_space, env.device, **common,
                         clip_actions=bool(v.get("clip_actions", False)))
        print("==================================================\nRecurrent models (PPO-GRU)\n"
              "==================================================\n%s\n%s\n--------------------------------------------------"
              % (policy, value), flush=True)
        return {"agent": {"policy": policy, "value": value}}


def describe(runner):
    """Architecture stamp for the run's sidecar."""
    models = runner.agent.models
    return {"variant": "gru_tanh" if getattr(runner, "_squash", False) else "gru",
            "action_distribution": ("tanh-transformed Gaussian (exact log-prob)" if getattr(runner, "_squash", False)
                                    else "yaml Gaussian with sample clipping"),
            "sequence_length": runner._sequence_length, "hidden_size": GRU_HIDDEN,
            "head_layers": list(HEAD_LAYERS), "agent_class": type(runner.agent).__name__,
            "parameters": {role: int(sum(p.numel() for p in model.parameters())) for role, model in models.items()},
            "models": {role: str(model) for role, model in models.items()}}
