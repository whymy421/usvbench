# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Acceptance tests for the tanh-squashed Gaussian SAC policy.

Pure torch on CPU: no Isaac, no gym.make, no environment. Run it with the
project python:

    C:/usvb/env/Scripts/python.exe tasks/_shared/test_squashed_gaussian.py

The comparison rows instantiate skrl's REAL ``gaussian_model`` (not a
stand-in), force both models to the same pre-squash mean and the same log-std,
and then read off the log-probability each one reports for the action it
actually returned.
"""

from __future__ import annotations

import math
import os

import gymnasium
import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, Model
from skrl.utils.model_instantiators.torch import gaussian_model

try:  # inside the repo package
    from .squashed_gaussian import MEAN_CLAMP, SquashedGaussianPolicy, squashed_gaussian_model
except ImportError:  # direct execution, or the flat wf_ copy on a box
    try:
        from squashed_gaussian import MEAN_CLAMP, SquashedGaussianPolicy, squashed_gaussian_model
    except ImportError:
        from wf_squash import MEAN_CLAMP, SquashedGaussianPolicy, squashed_gaussian_model

DEVICE = "cpu"
OBS_DIM = 12
ACT_DIM = 2
BOUND = 1e4  # the "finite and bounded" gate from the task
NETWORK = [{"name": "net", "input": "OBSERVATIONS", "layers": [128, 128, 64], "activations": "elu"}]


def obs_space(dim: int = OBS_DIM) -> gymnasium.spaces.Box:
    return gymnasium.spaces.Box(low=-float("inf"), high=float("inf"), shape=(dim,))


def act_space(low=-1.0, high=1.0, dim: int = ACT_DIM) -> gymnasium.spaces.Box:
    import numpy as np

    return gymnasium.spaces.Box(
        low=np.full((dim,), low, dtype=np.float32) if not isinstance(low, (list, tuple)) else np.asarray(low, dtype=np.float32),
        high=np.full((dim,), high, dtype=np.float32) if not isinstance(high, (list, tuple)) else np.asarray(high, dtype=np.float32),
        shape=(dim,),
    )


def force_constant_mean(model: nn.Module, value: float) -> None:
    """Pin the last Linear so the pre-squash mean is exactly ``value``.

    Works for both our model and skrl's generated one regardless of how the
    layers are named; the callers assert the resulting mean afterwards.
    """
    last = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last = module
    assert last is not None, "no nn.Linear found"
    # skrl's instantiator emits nn.LazyLinear, whose parameters do not exist
    # until the first forward pass; build_stock materialises them first.
    assert not isinstance(last.weight, nn.parameter.UninitializedParameter), (
        "lazy layer not materialised: run one forward pass before pinning"
    )
    with torch.no_grad():
        last.weight.zero_()
        last.bias.fill_(value)


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# --------------------------------------------------------------------------
# 1. extreme observations through a real (randomly initialised) network
# --------------------------------------------------------------------------
def test_extreme_observations() -> None:
    banner("1. extreme observations (+-1e3) through the MLP")
    torch.manual_seed(0)
    policy = SquashedGaussianPolicy(
        observation_space=obs_space(), action_space=act_space(), device=DEVICE, network=None
    )
    batch = 512
    states = torch.empty(batch, OBS_DIM).uniform_(-1.0, 1.0)
    states[: batch // 2] = 1e3 * torch.sign(torch.randn(batch // 2, OBS_DIM))
    states[batch // 2 :] *= 1e3

    actions, log_prob, outputs = policy.act({"states": states}, role="policy")

    assert actions.shape == (batch, ACT_DIM), actions.shape
    assert log_prob.shape == (batch, 1), log_prob.shape
    assert outputs["mean_actions"].shape == (batch, ACT_DIM)
    assert torch.isfinite(log_prob).all(), "non-finite log_prob"
    assert log_prob.abs().max().item() < BOUND, log_prob.abs().max().item()
    assert (actions.abs() < 1.0).all(), actions.abs().max().item()
    assert (outputs["mean_actions"].abs() < 1.0).all(), outputs["mean_actions"].abs().max().item()

    print(f"  states            |x| max = {states.abs().max().item():.1f}")
    print(f"  actions           range   = [{actions.min().item():+.8f}, {actions.max().item():+.8f}]")
    print(f"  mean_actions      range   = [{outputs['mean_actions'].min().item():+.8f}, "
          f"{outputs['mean_actions'].max().item():+.8f}]")
    print(f"  log_prob          range   = [{log_prob.min().item():+.4f}, {log_prob.max().item():+.4f}]")
    print(f"  log_prob finite           = {bool(torch.isfinite(log_prob).all())}")
    print("  PASS")


# --------------------------------------------------------------------------
# 2. extreme pre-squash means, swept
# --------------------------------------------------------------------------
def test_extreme_pre_squash_means() -> None:
    banner("2. extreme pre-squash means, swept")
    torch.manual_seed(1)
    states = torch.randn(256, OBS_DIM)
    print(f"  {'forced mean':>12} | {'log_prob min':>13} | {'log_prob max':>13} | {'|a| max':>12} | finite")
    print("  " + "-" * 70)
    for value in (1e1, 1e2, 1e3, 1e4, -1e4):
        policy = SquashedGaussianPolicy(
            observation_space=obs_space(), action_space=act_space(), device=DEVICE
        )
        force_constant_mean(policy, value)
        raw_mean, _, _ = policy.compute({"states": states})
        assert torch.allclose(raw_mean, torch.full_like(raw_mean, value)), "mean not pinned"

        actions, log_prob, outputs = policy.act({"states": states}, role="policy")
        finite = bool(torch.isfinite(log_prob).all())
        print(f"  {value:>12.0f} | {log_prob.min().item():>13.4f} | {log_prob.max().item():>13.4f} | "
              f"{actions.abs().max().item():>12.9f} | {finite}")
        assert finite, f"non-finite log_prob at mean={value}"
        assert log_prob.abs().max().item() < BOUND, f"|log_prob| >= {BOUND} at mean={value}"
        assert (actions.abs() < 1.0).all(), f"action on/outside bound at mean={value}"
        assert (outputs["mean_actions"].abs() < 1.0).all()
    print(f"  (soft mean clamp = {MEAN_CLAMP}, tanh({MEAN_CLAMP}) = {math.tanh(MEAN_CLAMP):.9f})")
    print("  PASS")


# --------------------------------------------------------------------------
# 3. the Jacobian correction is the RIGHT one
# --------------------------------------------------------------------------
def test_log_prob_matches_transformed_distribution() -> None:
    banner("3. log-prob equals torch's TanhTransform+AffineTransform reference")
    torch.manual_seed(2)
    space = act_space(low=[-2.0, -0.5], high=[2.0, 0.5])
    policy = SquashedGaussianPolicy(observation_space=obs_space(), action_space=space, device=DEVICE)
    states = torch.randn(64, OBS_DIM)

    actions, log_prob, _ = policy.act({"states": states}, role="policy")

    # rebuild the same distribution independently, straight from torch
    mean = policy._soft_clamp_mean(policy.net(states))
    std = policy.log_std_parameter.clamp(policy._g_log_std_min, policy._g_log_std_max).exp()
    reference = torch.distributions.TransformedDistribution(
        torch.distributions.Normal(mean, std),
        [
            torch.distributions.transforms.TanhTransform(cache_size=1),
            torch.distributions.transforms.AffineTransform(
                loc=policy.action_bias, scale=policy.action_scale
            ),
        ],
    )
    reference_log_prob = reference.log_prob(actions).sum(dim=-1, keepdim=True)
    gap = (log_prob - reference_log_prob).abs().max().item()

    print(f"  action space low/high     = {space.low.tolist()} / {space.high.tolist()}")
    print(f"  scale / bias              = {policy.action_scale.tolist()} / {policy.action_bias.tolist()}")
    print(f"  ours     log_prob[:3]     = {[round(v, 6) for v in log_prob[:3, 0].tolist()]}")
    print(f"  torch ref log_prob[:3]    = {[round(v, 6) for v in reference_log_prob[:3, 0].tolist()]}")
    print(f"  max abs difference        = {gap:.3e}")
    assert gap < 2e-3, gap
    assert (actions[:, 0].abs() < 2.0).all() and (actions[:, 1].abs() < 0.5).all(), "outside scaled bounds"
    print("  PASS")


# --------------------------------------------------------------------------
# 4. the stable Jacobian form actually matters
# --------------------------------------------------------------------------
def test_stable_vs_naive_jacobian() -> None:
    banner("4. stable log(1-tanh^2) vs the naive eps form")
    u = torch.tensor([0.0, 1.0, 5.0, 9.0, 15.0, 40.0], dtype=torch.float32)
    stable = SquashedGaussianPolicy._log_one_minus_tanh_sq(u)
    naive = torch.log(1.0 - torch.tanh(u) ** 2 + 1e-6)
    print(f"  {'u':>6} | {'stable':>14} | {'naive(eps=1e-6)':>16} | {'abs err':>12}")
    print("  " + "-" * 58)
    for i in range(u.numel()):
        print(f"  {u[i].item():>6.1f} | {stable[i].item():>14.6f} | {naive[i].item():>16.6f} | "
              f"{abs(stable[i].item() - naive[i].item()):>12.6f}")
    assert torch.isfinite(stable).all()
    assert abs(stable[-1].item() + 2 * (40.0 - math.log(2.0))) < 1e-3, stable[-1].item()
    assert abs(naive[-1].item() - math.log(1e-6)) < 1e-3, "naive form should have saturated at log(eps)"
    print("  PASS (naive saturates at log(1e-6) = -13.8155 and its gradient dies; stable stays exact)")


# --------------------------------------------------------------------------
# 5. gradients
# --------------------------------------------------------------------------
def test_gradients_flow() -> None:
    banner("5. gradient flow through the reparameterised sample")
    torch.manual_seed(3)
    policy = SquashedGaussianPolicy(observation_space=obs_space(), action_space=act_space(), device=DEVICE)
    states = 1e3 * torch.randn(128, OBS_DIM)  # extreme states here too

    _, log_prob, _ = policy.act({"states": states}, role="policy")
    loss = -log_prob.mean()
    loss.backward()

    nonzero = 0
    for name, param in policy.named_parameters():
        assert param.grad is not None, f"{name}: grad is None"
        assert torch.isfinite(param.grad).all(), f"{name}: non-finite grad"
        if param.grad.abs().max().item() > 0:
            nonzero += 1
    total = sum(1 for _ in policy.named_parameters())
    print(f"  loss (= -log_prob.mean()) = {loss.item():.6f}")
    print(f"  parameters with grad      = {total}/{total} non-None, all finite")
    print(f"  parameters with |grad|>0  = {nonzero}/{total}")
    print(f"  log_std grad              = {policy.log_std_parameter.grad.tolist()}")
    assert nonzero > 0, "every gradient was exactly zero"
    assert policy.log_std_parameter.grad.abs().max().item() > 0, "log_std received no gradient"
    print("  PASS")


# --------------------------------------------------------------------------
# 6. taken_actions branch round-trips
# --------------------------------------------------------------------------
def test_taken_actions_roundtrip() -> None:
    banner("6. taken_actions branch reproduces the sampled log-prob")
    torch.manual_seed(4)
    policy = SquashedGaussianPolicy(observation_space=obs_space(), action_space=act_space(), device=DEVICE)
    states = torch.randn(64, OBS_DIM)
    actions, log_prob, _ = policy.act({"states": states}, role="policy")
    _, log_prob_taken, _ = policy.act({"states": states, "taken_actions": actions}, role="policy")
    gap = (log_prob - log_prob_taken).abs().max().item()
    print(f"  sampled  log_prob[:3]     = {[round(v, 6) for v in log_prob[:3, 0].tolist()]}")
    print(f"  taken    log_prob[:3]     = {[round(v, 6) for v in log_prob_taken[:3, 0].tolist()]}")
    print(f"  max abs difference        = {gap:.3e}")
    assert gap < 1e-2, gap
    print("  PASS")


# --------------------------------------------------------------------------
# 7. head-to-head against skrl's stock GaussianMixin
# --------------------------------------------------------------------------
def build_stock(output: str, min_log_std: float, action_space) -> Model:
    """skrl's own instantiator -- the exact object the yaml would build.

    The generated container is built from ``nn.LazyLinear``, so one forward
    pass is needed before its weights exist and can be pinned.
    """
    model = gaussian_model(
        observation_space=obs_space(),
        action_space=action_space,
        device=DEVICE,
        clip_actions=True,
        clip_log_std=True,
        min_log_std=min_log_std,
        max_log_std=2.0,
        initial_log_std=0.0,
        network=NETWORK,
        output=output,
    )
    model.init_state_dict("policy")  # skrl's own lazy-module materialiser
    return model


def test_versus_stock_gaussian_mixin() -> None:
    banner("7. stock skrl GaussianMixin vs squashed, identical extreme means")
    space = act_space()
    torch.manual_seed(5)
    states = torch.randn(4096, OBS_DIM)
    alpha = 1.0  # SAC entropy coefficient; target term is -alpha * log_prob
    q_values = torch.zeros(4096, 1)  # a healthy critic, so the damage is all log_prob

    sources = {
        spec: gaussian_model(
            observation_space=obs_space(), action_space=space, device=DEVICE, clip_actions=True,
            clip_log_std=True, min_log_std=-2.0, max_log_std=2.0, initial_log_std=0.0,
            network=NETWORK, output=spec, return_source=True,
        )
        for spec in ("ACTIONS", "tanh(ACTIONS)")
    }
    # skrl emits "nn.functional.tanh(net)", not "torch.tanh" -- read, don't guess.
    assert "tanh" in sources["tanh(ACTIONS)"], sources["tanh(ACTIONS)"]
    assert "tanh" not in sources["ACTIONS"], sources["ACTIONS"]
    tanh_line = [ln.strip() for ln in sources["tanh(ACTIONS)"].splitlines() if "tanh" in ln][0]
    print(f"  skrl source, output='tanh(ACTIONS)' -> {tanh_line}")
    print("  skrl source, output='ACTIONS'        -> no tanh (mean is unbounded)")
    print(f"  {'model':<34} | {'mean':>7} | {'log_prob min':>15} | {'|a| max':>11} | {'on-bound':>8}")
    print("  " + "-" * 92)

    rows = []
    for mean_value in (10.0, 1e3, 1e4):
        # (a) stock, unbounded mean, clip_actions=True  -- the original defect
        stock = build_stock("ACTIONS", -20.0, space)
        force_constant_mean(stock, mean_value)
        with torch.no_grad():
            a_stock, lp_stock, _ = stock.act({"states": states}, role="policy")

        # (b) stock + the yaml v3 fix: tanh(ACTIONS) mean, min_log_std -2
        stock_tanh = build_stock("tanh(ACTIONS)", -2.0, space)
        force_constant_mean(stock_tanh, mean_value)
        with torch.no_grad():
            a_tanh, lp_tanh, _ = stock_tanh.act({"states": states}, role="policy")

        # the pins really took: unbounded mean vs tanh-bounded mean
        with torch.no_grad():
            mean_stock = stock.compute({"states": states})[0]
            mean_tanh = stock_tanh.compute({"states": states})[0]
        assert torch.allclose(mean_stock, torch.full_like(mean_stock, mean_value)), mean_stock[0]
        assert torch.allclose(mean_tanh, torch.full_like(mean_tanh, math.tanh(mean_value)), atol=1e-6)

        # (c) ours
        ours = SquashedGaussianPolicy(
            observation_space=obs_space(), action_space=space, device=DEVICE, min_log_std=-2.0
        )
        force_constant_mean(ours, mean_value)
        with torch.no_grad():
            a_ours, lp_ours, _ = ours.act({"states": states}, role="policy")

        for label, act_t, lp_t in (
            ("stock GaussianMixin (clip)", a_stock, lp_stock),
            ("stock + yaml tanh(ACTIONS)", a_tanh, lp_tanh),
            ("SquashedGaussianPolicy", a_ours, lp_ours),
        ):
            on_bound = (act_t.abs() >= 1.0).float().mean().item() * 100.0
            print(f"  {label:<34} | {mean_value:>7.0f} | {lp_t.min().item():>15.4e} | "
                  f"{act_t.abs().max().item():>11.8f} | {on_bound:>7.1f}%")
            rows.append((label, mean_value, lp_t, act_t, on_bound))
        print("  " + "-" * 92)

    # what SAC actually consumes:  target = min(q1,q2) - alpha * log_prob
    print("\n  SAC TD-target term  min(q1,q2) - alpha*log_prob  with q == 0, alpha == 1:")
    for label, mean_value, lp_t, _, _ in rows:
        target = q_values - alpha * lp_t
        print(f"    {label:<34} mean={mean_value:>7.0f}  target range = "
              f"[{target.min().item():+.4e}, {target.max().item():+.4e}]")

    stock_worst = min(lp.min().item() for label, _, lp, _, _ in rows if label.startswith("stock GaussianMixin"))
    tanh_worst = min(lp.min().item() for label, _, lp, _, _ in rows if label.startswith("stock + yaml"))
    ours_worst = max(lp.abs().max().item() for label, _, lp, _, _ in rows if label.startswith("Squashed"))
    tanh_atom = max(ob for label, _, _, _, ob in rows if label.startswith("stock + yaml"))

    print(f"\n  stock worst log_prob      = {stock_worst:.4e}")
    print(f"  stock+tanh worst log_prob = {tanh_worst:.4e}  "
          f"(bounded, but {tanh_atom:.1f}% of its samples sit EXACTLY on the bound:")
    print("                              clipping puts finite probability mass on a point, so the")
    print("                              reported density is not the density of the action taken)")
    print(f"  ours worst |log_prob|     = {ours_worst:.4e}")
    assert stock_worst < -1e6, f"stock should blow up, got {stock_worst}"
    assert ours_worst < BOUND, ours_worst
    assert tanh_atom > 0.0, "expected the clipped stock+tanh model to produce boundary atoms"
    print("  PASS")


# --------------------------------------------------------------------------
# 8. the Runner-facing instantiator
# --------------------------------------------------------------------------
def test_instantiator_matches_runner_contract() -> None:
    banner("8. squashed_gaussian_model(...) honours the Runner call convention")
    source = squashed_gaussian_model(
        observation_space=obs_space(), action_space=act_space(), device=DEVICE,
        clip_actions=True, clip_log_std=True, min_log_std=-2.0, max_log_std=2.0,
        initial_log_std=0.0, network=NETWORK, output="ACTIONS", return_source=True,
    )
    assert isinstance(source, str), type(source)
    print(f"  return_source=True  -> {source}")
    model = squashed_gaussian_model(
        observation_space=obs_space(), action_space=act_space(), device=DEVICE,
        clip_actions=True, clip_log_std=True, min_log_std=-2.0, max_log_std=2.0,
        initial_log_std=0.0, network=NETWORK, output="ACTIONS",
    )
    assert isinstance(model, Model) and isinstance(model, GaussianMixin)
    a, lp, out = model.act({"states": torch.randn(8, OBS_DIM)}, role="policy")
    print(f"  instance            -> {type(model).__name__}, act -> "
          f"actions{tuple(a.shape)}, log_prob{tuple(lp.shape)}, keys={sorted(out)}")
    assert "mean_actions" in out
    assert lp.shape == (8, 1)

    # The integration hook itself: Runner._component is a hard-coded whitelist,
    # so a yaml `class:` entry alone can never reach this model. Verify the
    # 4-line subclass that makes it reachable (no env needed: _component reads
    # no instance state, so an uninitialised Runner is enough to exercise it).
    from skrl.utils.runner.torch import Runner
    from skrl.utils.model_instantiators.torch import gaussian_model as stock_factory

    class SquashedRunner(Runner):
        def _component(self, name):
            if name.lower() == "squashedgaussianmixin":
                return squashed_gaussian_model
            return super()._component(name)

    stub = SquashedRunner.__new__(SquashedRunner)
    assert stub._component("SquashedGaussianMixin") is squashed_gaussian_model
    assert stub._component("GaussianMixin") is stock_factory  # untouched
    assert stub._component("DeterministicMixin") is not None  # critics untouched
    print("  Runner subclass: 'SquashedGaussianMixin' -> squashed_gaussian_model, "
          "'GaussianMixin'/'DeterministicMixin' unchanged")

    # exactly the call the Runner makes for the v3 policy block, output -> ACTIONS
    v3_block = {
        "clip_actions": True, "clip_log_std": True, "min_log_std": -2.0, "max_log_std": 2.0,
        "initial_log_std": 0.0, "network": NETWORK, "output": "ACTIONS",
    }
    v3_model = stub._component("SquashedGaussianMixin")(
        observation_space=obs_space(), action_space=act_space(), device=DEVICE, **v3_block
    )
    a3, lp3, out3 = v3_model.act({"states": torch.randn(16, OBS_DIM)}, role="policy")
    assert torch.isfinite(lp3).all() and (a3.abs() < 1.0).all()
    print(f"  v3 policy block via the hook -> actions{tuple(a3.shape)} in bounds, "
          f"log_prob{tuple(lp3.shape)} finite, mean_actions present={('mean_actions' in out3)}")

    for bad in ("tanh(ACTIONS)", "ONE"):
        try:
            squashed_gaussian_model(
                observation_space=obs_space(), action_space=act_space(), device=DEVICE,
                network=NETWORK, output=bad,
            )
        except ValueError as exc:
            print(f"  output={bad!r:<16} -> ValueError: {str(exc)[:60]}...")
        else:
            raise AssertionError(f"output={bad!r} should have raised")
    print("  PASS")


# --------------------------------------------------------------------------
# 9. the real SAC agent, on CPU, with no environment
# --------------------------------------------------------------------------
def build_sac_agent(policy: Model):
    """A real skrl SAC agent around ``policy``: real critics, real memory.

    No environment, no Isaac, and NOTHING written to disk: write_interval and
    checkpoint_interval are pinned to 0, which is the only path in
    ``Agent.init`` that creates neither a SummaryWriter nor a checkpoint dir.
    """
    import copy

    from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
    from skrl.memories.torch import RandomMemory
    from skrl.utils.model_instantiators.torch import deterministic_model

    obs, act = obs_space(), act_space()
    models = {"policy": policy}
    for role in ("critic_1", "critic_2", "target_critic_1", "target_critic_2"):
        critic = deterministic_model(
            observation_space=obs, action_space=act, device=DEVICE, clip_actions=False,
            network=[{"name": "net", "input": "OBSERVATIONS_ACTIONS", "layers": [64, 64], "activations": "elu"}],
            output="ONE",
        )
        critic.init_state_dict(role)
        models[role] = critic

    memory = RandomMemory(memory_size=64, num_envs=8, device=DEVICE)
    cfg = copy.deepcopy(SAC_DEFAULT_CONFIG)
    cfg["batch_size"] = 64
    cfg["gradient_steps"] = 1
    cfg["learning_starts"] = 0
    cfg["random_timesteps"] = 0
    cfg["discount_factor"] = 0.99
    cfg["experiment"]["write_interval"] = 0
    cfg["experiment"]["checkpoint_interval"] = 0
    cfg["experiment"]["directory"] = "wf_sac_smoke_never_written"

    agent = SAC(models=models, memory=memory, observation_space=obs, action_space=act, device=DEVICE, cfg=cfg)
    agent.init(trainer_cfg={"timesteps": 0})
    # Agent.init only assigns .writer inside the `write_interval > 0` branch,
    # and only mkdirs when checkpoint_interval > 0. Prove both stayed shut.
    assert getattr(agent, "writer", None) is None, "the smoke agent opened a TensorBoard writer"
    experiment_dir = getattr(agent, "experiment_dir", None)
    assert experiment_dir is None or not os.path.exists(experiment_dir), (
        f"the smoke agent created {experiment_dir}"
    )

    torch.manual_seed(9)
    for _ in range(64):  # fill the replay buffer with bounded, sane transitions
        memory.add_samples(
            states=torch.randn(8, OBS_DIM),
            actions=torch.empty(8, ACT_DIM).uniform_(-1.0, 1.0),
            rewards=0.01 * torch.randn(8, 1),
            next_states=torch.randn(8, OBS_DIM),
            terminated=torch.zeros(8, 1, dtype=torch.bool),
            truncated=torch.zeros(8, 1, dtype=torch.bool),
        )
    return agent


UPDATES = 400


def test_sac_agent_contract() -> None:
    banner(f"9. real skrl SAC agent (CPU, no env, {UPDATES} updates): does the TD target survive?")
    probe_states = torch.randn(256, OBS_DIM)
    probe_actions = torch.empty(256, ACT_DIM).uniform_(-1.0, 1.0)

    results = {}
    for label, make_policy in (
        ("stock GaussianMixin", lambda: build_stock("ACTIONS", -20.0, act_space())),
        ("SquashedGaussianPolicy", lambda: SquashedGaussianPolicy(
            observation_space=obs_space(), action_space=act_space(), device=DEVICE, min_log_std=-2.0)),
    ):
        torch.manual_seed(11)
        policy = make_policy()
        force_constant_mean(policy, 1e3)  # the runaway mean this class exists for
        agent = build_sac_agent(policy)

        # the TD target exactly as sac.py builds it, before any update
        with torch.no_grad():
            next_actions, next_log_prob, _ = agent.policy.act({"states": probe_states}, role="policy")
            q1, _, _ = agent.target_critic_1.act(
                {"states": probe_states, "taken_actions": next_actions}, role="target_critic_1")
            q2, _, _ = agent.target_critic_2.act(
                {"states": probe_states, "taken_actions": next_actions}, role="target_critic_2")
            target = torch.min(q1, q2) - agent._entropy_coefficient * next_log_prob

        for _ in range(UPDATES):
            agent._update(timestep=0, timesteps=1)

        with torch.no_grad():
            q_after, _, _ = agent.critic_1.act(
                {"states": probe_states, "taken_actions": probe_actions}, role="critic_1")
        finite_params = all(torch.isfinite(p).all() for p in agent.policy.parameters())
        results[label] = {
            "log_prob": (next_log_prob.min().item(), next_log_prob.max().item()),
            "target": (target.min().item(), target.max().item()),
            "q_after": q_after.abs().max().item(),
            "alpha": float(agent._entropy_coefficient),
            "finite": finite_params,
        }

    print(f"  {'policy':<24} | {'log_prob min':>13} | {'|TD target| max':>16} | "
          f"{'|Q| after upd':>16} | {'alpha':>9}")
    print("  " + "-" * 92)
    for label, r in results.items():
        print(f"  {label:<24} | {r['log_prob'][0]:>13.4e} | "
              f"{max(abs(r['target'][0]), abs(r['target'][1])):>16.4e} | {r['q_after']:>16.4e} | "
              f"{r['alpha']:>9.4f}")

    ours = results["SquashedGaussianPolicy"]
    stock = results["stock GaussianMixin"]
    assert ours["finite"], "policy parameters went non-finite"
    assert max(abs(v) for v in ours["target"]) < 1e3, ours["target"]
    assert ours["q_after"] < 1e3, ours["q_after"]
    assert max(abs(v) for v in stock["target"]) > 1e5, stock["target"]
    # the critic chases the poisoned target, and the entropy coefficient
    # collapses because -(log_alpha * (log_prob + target_entropy)) is dominated
    # by a log_prob of -1e6
    assert stock["q_after"] > 1e3, stock["q_after"]
    assert stock["alpha"] < ours["alpha"], (stock["alpha"], ours["alpha"])
    print(f"\n  critic chase:  stock |Q| {stock['q_after']:.4e} vs ours {ours['q_after']:.4e} "
          f"(both started near 0)")
    print(f"  entropy coeff: stock alpha {stock['alpha']:.4f} (collapsing from 0.2) vs "
          f"ours {ours['alpha']:.4f}")
    print(f"\n  stock TD target is {max(abs(v) for v in stock['target']) / max(abs(v) for v in ours['target']):.3e}x "
          "larger than ours, from the log-prob term alone")
    print("  PASS")


def main() -> None:
    print(f"torch {torch.__version__}  device {DEVICE}")
    test_extreme_observations()
    test_extreme_pre_squash_means()
    test_log_prob_matches_transformed_distribution()
    test_stable_vs_naive_jacobian()
    test_gradients_flow()
    test_taken_actions_roundtrip()
    test_versus_stock_gaussian_mixin()
    test_instantiator_matches_runner_contract()
    test_sac_agent_contract()
    print("\n" + "=" * 72)
    print("ALL TESTS PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()
