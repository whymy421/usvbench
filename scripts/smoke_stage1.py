"""Harbor Stage1 smoke: 49-D obs, finite rewards, config wiring, clean reset.

Termination-on-milestone cannot be provoked by random actions from the empty
berth (no obstacles near spawn, gates need directed crossings), so it is
validated post-training from episode-length telemetry instead of here.
"""
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg("Isaac-USV-HarborStage1-Direct-v1", device="cuda:0", num_envs=8)
cfg.seed = 42
assert cfg.mission_depth == 1, cfg.mission_depth
assert cfg.terminate_on_milestone and cfg.terminate_on_contact
assert cfg.obs_kinematic and cfg.observation_space == 49
assert abs(cfg.episode_length_s - 60.0) < 1e-6

env = gym.make("Isaac-USV-HarborStage1-Direct-v1", cfg=cfg)
base = env.unwrapped
obs, _ = env.reset()
o = obs["policy"]
assert o.shape == (8, 49), f"obs shape {o.shape}"
assert torch.isfinite(o).all()
# Kinematic channels sit at [3:6]; at spawn the boat is at rest.
assert o[:, 3:6].abs().max() < 0.2, f"kinematic channels not at rest: {o[:, 3:6]}"

horizon = int(base.max_episode_length)
rng = torch.Generator(device="cpu").manual_seed(0)
truncs = 0
for t in range(horizon + 5):
    a = (torch.rand((8, 2), generator=rng) * 2 - 1).to(o.device)
    obs, r, term, trunc, _ = env.step(a)
    assert torch.isfinite(r).all(), f"non-finite reward at {t}"
    assert r.abs().max() < 130.0, f"reward spike {r.abs().max()} at {t}"
    truncs += int(trunc.sum())
o2 = obs["policy"]
assert torch.isfinite(o2).all()
assert truncs >= 8, f"expected a full timeout round, saw {truncs} truncations"
print(f"timeout truncations: {truncs}", flush=True)
print("SMOKE STAGE1 PASS", flush=True)
env.close()
app.close()
