# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tanh-squashed Gaussian policy for skrl's SAC.

Why this file exists
--------------------
skrl 1.4.3 ships no squashed-Gaussian model. Its ``GaussianMixin.act`` (see
``skrl/models/torch/gaussian.py``) does exactly this:

    self._g_distribution = Normal(mean_actions, log_std.exp())
    actions = self._g_distribution.rsample()
    if self._g_clip_actions:
        actions = torch.clamp(actions, min=..., max=...)
    log_prob = self._g_distribution.log_prob(inputs.get("taken_actions", actions))

The mean is UNBOUNDED and ``clip_actions`` clamps the *sample* without
correcting the probability density. So once the mean runs away, every sample
lands on the action bound and the log-probability of that bound under the
runaway Normal is astronomically negative -- this project measured
``policy.log_prob`` minima of -9.8e15 while every other TD-target term stayed
bounded. SAC then multiplies that by the entropy coefficient inside

    target_q = min(q1, q2) - alpha * next_log_prob          (sac.py, _update)

and the critic target is destroyed.

The yaml-level mitigation (``output: tanh(ACTIONS)`` plus ``min_log_std: -2``)
bounds the *mean* and floors sigma, but the distribution is still an unsquashed
Normal: the reported log-probability is the density of the pre-squash variable,
not of the action actually taken. This module is the code-level fix -- a real
change-of-variables.

What it does
------------
    u ~ Normal(mu, sigma)                      (reparameterised, rsample)
    a = scale * tanh(u) + bias                 (bounded by construction)
    log p(a) = log p(u) - sum_i log|da_i/du_i|
             = log p(u) - sum_i [ log(scale_i) + log(1 - tanh(u_i)^2) ]

with the Jacobian term evaluated in the numerically stable form

    log(1 - tanh(u)^2) = 2 * (log 2 - u - softplus(-2u))

instead of the naive ``log(1 - tanh(u)**2 + eps)``, which underflows to
``log(eps)`` (a constant, hence a wrong gradient) as soon as |u| > ~9 in fp32.

Two extra guards, both deliberate and both documented because they are not part
of the textbook derivation:

1. ``mean_clamp``: the pre-squash mean is soft-clamped as
   ``c * tanh(mu / c)``. A hard ``clamp`` would zero the gradient once
   saturated; the soft form keeps it finite and non-zero. Without any clamp the
   correction term grows like ``2*|u|``, so a runaway mean of 1e4 would produce
   a perfectly finite but useless log-prob of ~2e4 per dimension. With c = 8,
   ``tanh(8) = 0.99999977`` -- the full action range is still reachable.
2. ``TANH_EPS``: in fp32 ``tanh(u)`` saturates to exactly +-1 for |u| > ~9, so
   the returned action is shrunk to the open interval. This keeps actions
   strictly interior; the Jacobian term still uses the analytic unsaturated
   form.

``outputs["mean_actions"]`` is the SQUASHED, scaled mean. This matters beyond
cosmetics: every evaluator in this repo (``eval_v6_frozen.py``,
``eval_benchmark.py``, ``eval_hazard_nav.py``, ... 14 call sites) takes the
deterministic action as ``outputs[-1].get("mean_actions", outputs[0])``, and so
does skrl's own trainer eval path. Returning the raw pre-squash mean there
would feed an unbounded vector to the environment at evaluation time.

Not wired into any run. See ``squashed_gaussian_model`` for the integration
hook: skrl's ``Runner._component`` is a hard-coded whitelist of class names, so
a yaml ``class:`` entry alone cannot select this model.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import gymnasium
import torch
import torch.nn as nn

from skrl import logger
from skrl.models.torch import GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space

# Defaults. min_log_std -5 floors sigma at 6.7e-3 (skrl's own default of -20
# floors it at 2e-9, which makes the pre-squash log-density huge on its own).
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
MEAN_CLAMP = 8.0
TANH_EPS = 1e-6

_ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "selu": nn.SELU,
    "silu": nn.SiLU,
    "sigmoid": nn.Sigmoid,
    "identity": nn.Identity,
}


def _build_mlp(num_in: int, hidden: Sequence[int], num_out: int, activation: str) -> nn.Sequential:
    """Plain MLP matching the yaml ``layers``/``activations`` block."""
    key = str(activation).lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"Unsupported activation {activation!r}; known: {sorted(_ACTIVATIONS)}")
    act = _ACTIVATIONS[key]
    layers: list[nn.Module] = []
    size = num_in
    for width in hidden:
        layers.append(nn.Linear(size, int(width)))
        layers.append(act())
        size = int(width)
    layers.append(nn.Linear(size, num_out))
    return nn.Sequential(*layers)


def _action_scale_and_bias(action_space: Any, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Affine map from tanh's (-1, 1) onto the action space.

    Isaac Lab tasks in this repo declare ``Box(low=-1.0, high=1.0)``, which
    gives scale 1 / bias 0. Unbounded Box spaces (Isaac's default when the cfg
    passes a bare int) have no meaningful scale, so fall back to the identity
    and say so rather than silently producing inf.
    """
    if isinstance(action_space, gymnasium.spaces.Box):
        low = torch.as_tensor(action_space.low, dtype=torch.float32, device=device).flatten()
        high = torch.as_tensor(action_space.high, dtype=torch.float32, device=device).flatten()
        if torch.isfinite(low).all() and torch.isfinite(high).all():
            return (high - low) / 2.0, (high + low) / 2.0
        logger.warning(
            "SquashedGaussianPolicy: action space has non-finite bounds; "
            "falling back to tanh output in (-1, 1)"
        )
    return torch.ones(1, dtype=torch.float32, device=device), torch.zeros(1, dtype=torch.float32, device=device)


class SquashedGaussianPolicy(GaussianMixin, Model):
    """SAC-ready tanh-squashed Gaussian policy.

    Mixin first, ``Model`` second -- skrl's ``Model.act`` logs
    "Make sure to place Mixins before Model during model definition" and raises
    if the MRO is the other way round.

    :param observation_space: passed straight to ``Model.__init__``
    :param action_space: ditto; a finite ``Box`` also supplies the output scale
    :param device: ``None`` resolves through ``skrl.config.torch.parse_device``
    :param clip_actions: accepted for config compatibility and IGNORED --
        sample clipping is exactly the bug this class removes; tanh bounds the
        action instead
    :param clip_log_std, min_log_std, max_log_std, reduction: as GaussianMixin
    :param initial_log_std: initial value of the state-independent log-std
    :param fixed_log_std: freeze the log-std parameter (no gradient)
    :param hidden_sizes, activation: MLP shape when ``network`` is not given
    :param network: pre-built torch module mapping observations -> mean actions
    :param mean_clamp: soft bound on the pre-squash mean (see module docstring)
    """

    def __init__(
        self,
        observation_space: Optional[Any] = None,
        action_space: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = LOG_STD_MIN,
        max_log_std: float = LOG_STD_MAX,
        reduction: str = "sum",
        initial_log_std: float = 0.0,
        fixed_log_std: bool = False,
        hidden_sizes: Sequence[int] = (128, 128, 64),
        activation: str = "elu",
        network: Optional[nn.Module] = None,
        mean_clamp: float = MEAN_CLAMP,
        role: str = "",
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        # clip_actions=False on purpose: GaussianMixin's clamp is the defect.
        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
            role=role,
        )
        if clip_actions:
            logger.warning(
                "SquashedGaussianPolicy: clip_actions=True ignored; the tanh squash bounds actions "
                "and clipping samples is what breaks the log-probability"
            )
        if mean_clamp <= 0.0:
            raise ValueError(f"mean_clamp must be positive, got {mean_clamp}")

        self._mean_clamp = float(mean_clamp)
        self.net = network if network is not None else _build_mlp(
            self.num_observations, hidden_sizes, self.num_actions, activation
        )
        self.log_std_parameter = nn.Parameter(
            torch.full((self.num_actions,), float(initial_log_std)), requires_grad=not fixed_log_std
        )

        scale, bias = _action_scale_and_bias(action_space, self.device)
        # Buffers, so .to(device) and state_dict() follow the model.
        self.register_buffer("action_scale", scale)
        self.register_buffer("action_bias", bias)
        self.register_buffer("log_action_scale", torch.log(scale))

    # ------------------------------------------------------------------ maths

    def _soft_clamp_mean(self, mean_actions: torch.Tensor) -> torch.Tensor:
        c = self._mean_clamp
        return c * torch.tanh(mean_actions / c)

    @staticmethod
    def _log_one_minus_tanh_sq(pre_actions: torch.Tensor) -> torch.Tensor:
        """log(1 - tanh(u)^2), stable for |u| up to fp32 range.

        Naive form: log(1 - tanh(u)**2 + eps) -> log(eps) for |u| > 9 (fp32),
        i.e. a constant with zero gradient. This form stays exact.
        """
        return 2.0 * (math.log(2.0) - pre_actions - torch.nn.functional.softplus(-2.0 * pre_actions))

    def _squash(self, pre_actions: torch.Tensor) -> torch.Tensor:
        """tanh + affine map, shrunk into the OPEN interval (see TANH_EPS)."""
        squashed = torch.clamp(torch.tanh(pre_actions), -1.0 + TANH_EPS, 1.0 - TANH_EPS)
        return squashed * self.action_scale + self.action_bias

    def _unsquash(self, actions: torch.Tensor) -> torch.Tensor:
        """Inverse map, for the ``taken_actions`` branch."""
        normalized = (actions - self.action_bias) / self.action_scale
        normalized = torch.clamp(normalized, -1.0 + TANH_EPS, 1.0 - TANH_EPS)
        return torch.atanh(normalized)

    # ------------------------------------------------------------------- skrl

    def compute(
        self, inputs: Mapping[str, Union[torch.Tensor, Any]], role: str = ""
    ) -> Tuple[torch.Tensor, torch.Tensor, Mapping[str, Union[torch.Tensor, Any]]]:
        """Observations -> (pre-squash mean, log-std, extras).

        Raw network output; ``act`` applies the soft mean clamp so the guard
        survives a subclass that overrides this method. The unflatten call is
        what skrl's own generated models do (identity for a flat Box space).
        """
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        return self.net(states), self.log_std_parameter, {}

    def act(
        self, inputs: Mapping[str, Union[torch.Tensor, Any]], role: str = ""
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        """Same contract as ``GaussianMixin.act``: (actions, log_prob, outputs).

        ``log_prob`` has shape (N, 1) under the default "sum" reduction, which
        is what SAC's ``min(q1, q2) - alpha * log_prob`` needs to broadcast
        against the (N, 1) critic outputs. ``outputs["mean_actions"]`` is the
        squashed deterministic action.
        """
        mean_actions, log_std, outputs = self.compute(inputs, role)

        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, self._g_log_std_min, self._g_log_std_max)
        self._g_log_std = log_std
        self._g_num_samples = mean_actions.shape[0]

        mean_actions = self._soft_clamp_mean(mean_actions)
        self._g_distribution = torch.distributions.Normal(mean_actions, log_std.exp())

        taken_actions = inputs.get("taken_actions", None)
        if taken_actions is None:
            pre_actions = self._g_distribution.rsample()  # reparameterisation trick
        else:
            pre_actions = self._unsquash(taken_actions)

        # change of variables: log p(a) = log p(u) - sum log|da/du|
        log_prob = self._g_distribution.log_prob(pre_actions)
        log_prob = log_prob - self._log_one_minus_tanh_sq(pre_actions) - self.log_action_scale

        if self._g_reduction is not None:
            log_prob = self._g_reduction(log_prob, dim=-1)

        actions = self._squash(pre_actions)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        outputs["mean_actions"] = self._squash(mean_actions)
        return actions, log_prob, outputs

    def get_entropy(self, role: str = "") -> torch.Tensor:
        """Entropy of the PRE-SQUASH Normal (an upper bound on the squashed one).

        SAC never calls this -- it uses -log_prob as its entropy estimate -- but
        skrl's on-policy agents do, so the inherited behaviour is kept and
        labelled rather than silently reinterpreted.
        """
        return GaussianMixin.get_entropy(self, role)


def squashed_gaussian_model(
    observation_space: Optional[Any] = None,
    action_space: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = None,
    clip_actions: bool = False,
    clip_log_std: bool = True,
    min_log_std: float = LOG_STD_MIN,
    max_log_std: float = LOG_STD_MAX,
    reduction: str = "sum",
    initial_log_std: float = 0.0,
    fixed_log_std: bool = False,
    network: Sequence[Mapping[str, Any]] = (),
    output: Union[str, Sequence[str]] = "",
    mean_clamp: float = MEAN_CLAMP,
    return_source: bool = False,
    *args,
    **kwargs,
) -> Union[Model, str]:
    """Instantiator with skrl's ``gaussian_model`` signature, for the Runner.

    ``Runner._generate_models`` calls the instantiator twice: once with
    ``return_source=True`` (it prints the string) and once for real. Matching
    that signature is what lets a one-line hook in ``scripts/sac_train.py``
    swap this in for skrl's ``gaussian_model`` without touching the yaml
    schema.

    Only the network block shape this repo actually uses is parsed
    (``input: OBSERVATIONS``, ``layers: [...]``, ``activations: <name>``);
    anything else raises instead of being quietly reinterpreted.
    """
    blocks = list(network)
    if len(blocks) != 1:
        raise ValueError(f"squashed_gaussian_model expects exactly one network block, got {len(blocks)}")
    block = blocks[0]
    source = str(block.get("input", "OBSERVATIONS")).strip().upper()
    if source not in ("OBSERVATIONS", "STATES"):
        raise ValueError(f"squashed_gaussian_model only supports input OBSERVATIONS/STATES, got {block.get('input')!r}")
    hidden = [int(width) for width in block.get("layers", [])]
    activations = block.get("activations", "elu")
    if isinstance(activations, (list, tuple)):
        if len(set(activations)) != 1:
            raise ValueError(f"squashed_gaussian_model needs one activation for all layers, got {activations}")
        activations = activations[0]
    out_spec = str(output).strip().upper().replace(" ", "")
    if out_spec not in ("", "ACTIONS"):
        raise ValueError(
            f"output {output!r} is not valid here: this model squashes internally, so a yaml "
            "tanh(ACTIONS) would squash twice. Use 'ACTIONS' or omit it."
        )

    if return_source:
        return (
            "SquashedGaussianPolicy(net=MLP{hidden}, activation={act}, tanh-squashed with "
            "Jacobian-corrected log-prob, log_std in [{lo}, {hi}], mean_clamp={mc})".format(
                hidden=hidden, act=activations, lo=min_log_std, hi=max_log_std, mc=mean_clamp
            )
        )

    return SquashedGaussianPolicy(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        clip_actions=clip_actions,
        clip_log_std=clip_log_std,
        min_log_std=min_log_std,
        max_log_std=max_log_std,
        reduction=reduction,
        initial_log_std=initial_log_std,
        fixed_log_std=fixed_log_std,
        hidden_sizes=hidden,
        activation=activations,
        mean_clamp=mean_clamp,
    )
