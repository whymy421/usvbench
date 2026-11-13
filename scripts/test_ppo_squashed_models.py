"""Offline checks of the bounded tanh-Gaussian PPO policies (scripts/ppo_squashed_models.py). CPU only, no simulator.

usage: python -B scripts/test_ppo_squashed_models.py      (or pytest)
"""
from __future__ import annotations

import glob
import math
import os
import sys

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppo_memory_models as PM  # noqa: E402
import ppo_squashed_models as PS  # noqa: E402

OBS = gym.spaces.Box(-np.inf, np.inf, (42,))
ACT = gym.spaces.Box(-1.0, 1.0, (2,))
RUNS = r"C:\Users\Yutong\Desktop\Experiments\usvbench_ppo_fix_runs"


def shared(log_std=-1.9):
    torch.manual_seed(0)
    return PS.SharedTanhActorCritic(OBS, ACT, "cpu", initial_log_std=log_std)


def test_density_integrates_to_one():
    """1-D marginal of the transformed density over (-1, 1), for centred, off-centre and saturated latents."""
    space = gym.spaces.Box(-1.0, 1.0, (1,))
    for mu, log_std in ((0.0, -1.9), (2.5, -1.9), (-6.0, -0.7), (0.3, 0.5)):
        model = PS.SharedTanhActorCritic(OBS, space, "cpu", initial_log_std=log_std)
        with torch.no_grad():
            model.policy_layer.weight.zero_()
            model.policy_layer.bias.fill_(mu)
        # integrate in the latent variable (exact change of variables, no endpoint singularity): u in mu +- 12 sigma
        sigma = math.exp(log_std)
        u = torch.linspace(mu - 12 * sigma, mu + 12 * sigma, 200001).unsqueeze(-1)
        a = torch.tanh(u)
        keep = a.abs().squeeze(-1) < 1.0 - 1e-6           # the model's open interval
        states = torch.zeros(int(keep.sum()), 42)
        with torch.no_grad():
            _, logp, _ = model.act({"states": states, "taken_actions": a[keep]}, role="policy")
        da_du = 1.0 - a[keep].squeeze(-1) ** 2
        integral = torch.trapezoid(torch.exp(logp.squeeze(-1)) * da_du, u[keep].squeeze(-1)).item()
        assert abs(integral - 1.0) < 2e-3 or abs(mu) > 3, (mu, log_std, integral)   # saturated: mass beyond eps
        print("  density mu=%+.1f log_std=%+.1f: integral %.5f" % (mu, log_std, integral))


def test_collection_equals_replay():
    model = shared()
    states = torch.randn(4096, 42) * 3.0
    with torch.no_grad():
        model.policy_layer.bias.fill_(3.0)                # push some means far out: the clipped-Gaussian failure
        a, logp, out = model.act({"states": states}, role="policy")
        _, logp2, _ = model.act({"states": states, "taken_actions": a}, role="policy")
    assert torch.isfinite(logp).all() and torch.isfinite(logp2).all()
    assert (a.abs() < 1.0).all() and (out["mean_actions"].abs() < 1.0).all()
    err = (logp - logp2).abs().max().item()
    assert err < 1e-4, err
    print("  collection vs replay: max |dlogp| %.2e, min logp %.1f" % (err, logp.min().item()))


def test_entropy_estimate_matches_monte_carlo():
    model = shared(log_std=-0.5)
    states = torch.randn(8, 42)
    with torch.no_grad():
        model.act({"states": states}, role="policy")
        est = torch.stack([model.get_entropy().sum(dim=-1) for _ in range(4000)]).mean(0)
        mc = []
        for _ in range(4000):
            a, logp, _ = model.act({"states": states}, role="policy")
            mc.append(-logp.squeeze(-1))
        mc = torch.stack(mc).mean(0)
    assert (est - mc).abs().max().item() < 0.05, (est, mc)
    print("  entropy estimate vs Monte Carlo -E[log p(a)]: max diff %.3f" % (est - mc).abs().max().item())


def test_shared_value_path():
    model = shared()
    states = torch.randn(16, 42)
    with torch.no_grad():
        direct = model.value_layer(model.net_container(states))
        model.act({"states": states}, role="policy")
        cached, _, _ = model.act({"states": states}, role="value")
        fresh, _, _ = model.act({"states": states * 2.0}, role="value")      # no policy call before: recomputed
    assert torch.allclose(direct, cached)
    assert torch.allclose(fresh, model.value_layer(model.net_container(states * 2.0)))


def test_loads_the_release_mlp_checkpoint():
    """Same parameter names and shapes as the yaml's shared model: a release checkpoint loads without gaps."""
    paths = sorted(glob.glob(os.path.join(RUNS, "train", "tr_fortress_blueboat_geo_time_s90001", "final.pt")))
    if not paths:
        print("  (no release checkpoint found; skipped)")
        return
    state = torch.load(paths[0], map_location="cpu", weights_only=False)["policy"]
    missing, unexpected = shared().load_state_dict(state, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    print("  release MLP checkpoint loads into SharedTanhActorCritic: 0 missing, 0 unexpected keys")


def test_gru_tanh_train_path_equals_act_path():
    torch.manual_seed(0)
    N, L = 8, 32
    policy = PM._gru_tanh_policy_class()(OBS, ACT, "cpu", num_envs=N, sequence_length=L, initial_log_std=-1.9)
    with torch.no_grad():
        policy.gru.weight_hh_l0.mul_(3.0)
    h = torch.zeros(1, N, 128)
    for rollout in range(2):
        states, dones = torch.randn(L, N, 42), torch.rand(L, N) < 0.15
        stored, acts, logps = [], [], []
        policy.eval()
        with torch.no_grad():
            for t in range(L):
                stored.append(h.clone())
                a, logp, out = policy.act({"states": states[t], "rnn": [h]}, role="policy")
                acts.append(a)
                logps.append(logp)
                h = out["rnn"][0].clone()
                h[:, dones[t]] = 0.0
        seq = lambda x: x.permute(1, 0, *range(2, x.dim())).reshape(N * L, *x.shape[2:])  # noqa: E731
        s_seq, a_seq, lp_seq = seq(states), seq(torch.stack(acts)), seq(torch.stack(logps))
        h_seq = torch.stack(stored)[:, 0].permute(1, 0, 2).reshape(N * L, 128).unsqueeze(0)
        d_seq = dones.permute(1, 0).reshape(N * L, 1)
        policy.train()
        with torch.no_grad():
            _, lp_train, _ = policy.act({"states": s_seq, "taken_actions": a_seq, "rnn": [h_seq],
                                         "terminated": d_seq}, role="policy")
        err = (lp_train - lp_seq).abs().max().item()
        assert err < 1e-4, err
        print("  GRU tanh rollout %d: %d episode ends, max |logp_train - logp_act| %.2e" % (rollout, int(dones.sum()), err))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name)
            fn()
    print("ALL PASSED")
