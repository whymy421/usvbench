"""Offline checks of scripts/ppo_demo_warmstart.py (round 6): a fake env whose expert is a known function of the
observation (MLP case) or of the previous observation (needs memory: GRU case); imitation must learn it."""
import sys
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from skrl.resources.preprocessors.torch import RunningStandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ppo_demo_warmstart as DW  # noqa: E402
from ppo_memory_models import GRUValue, _gru_tanh_policy_class  # noqa: E402
from ppo_squashed_models import SharedTanhActorCritic  # noqa: E402

N, FDIM, ADIM, EP = 16, 6, 2, 50
obs_space = gym.spaces.Box(-np.inf, np.inf, (FDIM,))
act_space = gym.spaces.Box(-1.0, 1.0, (ADIM,))
W = torch.randn(FDIM, ADIM)


class Env:
    def __init__(self, memory):
        self.memory, self.t = memory, torch.zeros(N, dtype=torch.long)
        self.obs = torch.randn(N, FDIM) * 3 + 1
        self.prev = torch.zeros(N, FDIM)
        self.episode_success = torch.zeros(N, dtype=torch.bool)

    def reset(self):
        return self.obs.clone(), {}

    def step(self, a):
        self.t += 1
        done = self.t % EP == 0
        self.episode_success = done & (torch.arange(N) % 2 == 0)
        self.prev = torch.where(done.unsqueeze(1), torch.zeros_like(self.obs), self.obs)
        self.obs = torch.randn(N, FDIM) * 3 + 1
        rew = torch.where(done, torch.full((N,), 5.0), torch.full((N,), -0.1))
        return self.obs.clone(), rew.view(-1, 1), done.view(-1, 1), torch.zeros(N, 1, dtype=torch.bool), {}


class Expert:
    def __init__(self, env):
        self.env = env

    def act(self):
        src = self.env.prev if self.env.memory else self.env.obs
        return torch.tanh(src @ W / 3)

    def reset(self, done):
        pass


class Penalty:
    def __init__(self):
        self.values = None

    def refresh(self, rows=None):
        self.values = [0.0] * N

    def penalties(self, dones, codes):
        return {i: -10.0 for i, d in enumerate(dones) if d and i % 2}

    def summary(self):
        return {}


class Agent:
    pass


def make_agent(recurrent):
    agent = Agent()
    if recurrent:
        P = _gru_tanh_policy_class()
        pol = P(obs_space, act_space, "cpu", num_envs=N, sequence_length=32, initial_log_std=-1.9)
        val = GRUValue(obs_space, act_space, "cpu", num_envs=N, sequence_length=32)
        agent._rnn_initial_states = {"policy": [torch.ones(1, N, 128)], "value": [torch.ones(1, N, 128)]}
    else:
        pol = val = SharedTanhActorCritic(obs_space, act_space, "cpu")
    agent.models = {"policy": pol, "value": val}
    agent._state_preprocessor = RunningStandardScaler(size=FDIM, device="cpu")
    agent._value_preprocessor = RunningStandardScaler(size=1, device="cpu")
    agent.checkpoint_modules = {"policy": pol, "value": val, "state_preprocessor": agent._state_preprocessor,
                                "value_preprocessor": agent._value_preprocessor}
    return agent


def codes(env):
    return [3 if bool(s) else 2 for s in env.episode_success]


def run(recurrent, memory):
    torch.manual_seed(0)
    env = Env(memory)
    agent = make_agent(recurrent)
    log_std0 = agent.models["policy"].log_std_parameter.detach().clone()
    data, stats = DW.collect(env.step, env.reset, env, Expert(env), Penalty(), lambda: codes(env), 400, -1.0)
    assert stats["episodes_finished"] == N * (400 // EP), stats
    assert stats["successes"] == (N // 2) * (400 // EP), stats
    cfg = dict(DW.DEFAULTS, epochs_mlp=40, epochs_gru=12, batch=512, tail_steps=100, window=25)
    fit = DW.fit(agent, data, 0.99, cfg, recurrent, log=lambda m: None)
    assert torch.equal(agent.models["policy"].log_std_parameter.detach(), log_std0), "log_std must not move"
    var = fit["action_variance"]
    print("recurrent=%s memory=%s action mse %.4f (variance %.4f) value samples %d"
          % (recurrent, memory, fit["final_action_mse"], var, fit["value_samples"]))
    with tempfile.TemporaryDirectory() as d:
        path = DW.save_agent(agent, Path(d) / DW.BC_CHECKPOINT)
        state = torch.load(path, weights_only=False)
        assert set(state) == set(agent.checkpoint_modules)
    return fit["final_action_mse"] / var


def test_returns():
    rew = torch.tensor([[1.0], [1.0], [1.0], [1.0]])
    done = torch.tensor([[False], [True], [False], [False]])
    g, valid = DW.returns(rew, done, 0.5, tail_steps=2)
    assert torch.allclose(g[:, 0], torch.tensor([1.5, 1.0, 1.5, 1.0])), g
    assert valid[:, 0].tolist() == [True, True, True, False], valid   # t=2 has 2 steps left (>= tail), t=3 has 1


test_returns()
r_mlp = run(False, False)
assert r_mlp < 0.05, r_mlp                      # memoryless expert: the MLP imitates it
r_gru_mem = run(True, True)
assert r_gru_mem < 0.15, r_gru_mem              # expert reads the PREVIOUS observation: needs the GRU's memory
r_mlp_mem = run(False, True)
assert r_mlp_mem > 0.5, r_mlp_mem               # ... which the MLP cannot do
print("ALL OK (relative mse: mlp %.3f, gru on memory task %.3f, mlp on memory task %.3f)" % (r_mlp, r_gru_mem, r_mlp_mem))


# ------------------------------------------------------------------ round 7: anchor
def test_anchor(recurrent, memory):
    import ppo_training_fixes as FX
    torch.manual_seed(1)
    env = Env(memory)
    agent = make_agent(recurrent)
    data, _ = DW.collect(env.step, env.reset, env, Expert(env), Penalty(), lambda: codes(env), 400, -1.0)
    cfg = dict(DW.DEFAULTS, epochs_mlp=40, epochs_gru=12, batch=512, tail_steps=100, window=25)
    DW.fit(agent, data, 0.99, cfg, recurrent, log=lambda m: None)
    freeze = FX.NormaliserFreeze(10_000)
    freeze.attach(agent)
    calls = []
    agent._update = lambda *a, **k: calls.append(1)
    clock = {"t": 0}
    acfg = dict(DW.ANCHOR_DEFAULTS, lr=1e-3, batch_mlp=512, window_gru=50, burn_in_gru=25)
    with tempfile.TemporaryDirectory() as d:
        anchor = DW.install_anchor(agent, data, acfg, 1000, lambda: clock["t"], recurrent, freeze, d)
        assert freeze.frozen and freeze.frozen_at == 0
        digest = FX.module_digest(agent._state_preprocessor)
        with torch.no_grad():                                         # "PPO damage": perturb the policy
            for q in anchor.params:
                q.add_(0.3 * torch.randn_like(q))
        before = float(anchor._loss())
        for k in range(60):
            clock["t"] = k * 5
            agent._update()
        after = float(anchor._loss())
        assert len(calls) == 60 and anchor.updates == 60
        assert after < 0.5 * before, (before, after)                  # the anchor pulls the policy back
        assert FX.module_digest(agent._state_preprocessor) == digest  # the normaliser stayed frozen
        clock["t"] = 1000
        n0 = anchor.steps
        agent._update()
        assert anchor.steps == n0, "weight 0 at the end of the budget: no imitation step"
        assert (Path(d) / "demo_anchor.json").is_file()
        DW.save_data(data, Path(d) / DW.DEMO_DATA)
        back = DW.load_data(Path(d) / DW.DEMO_DATA)
        assert all(torch.equal(back[k], data[k]) for k in data)
    print("anchor recurrent=%s: bc mse %.4f -> %.4f after 60 anchored updates" % (recurrent, before, after))


test_anchor(False, False)
test_anchor(True, True)
print("ANCHOR OK")
