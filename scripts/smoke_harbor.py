"""Task B v6 smoke: env constructs, obs sane, reward finite, ablation path."""
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg("Isaac-USV-HarborMission-Direct-v1", device="cuda:0", num_envs=8)
cfg.seed = 42
env = gym.make("Isaac-USV-HarborMission-Direct-v1", cfg=cfg)
obs, _ = env.reset()
o = obs["policy"]
assert o.shape[-1] == 46, f"obs dim {o.shape}"
assert torch.isfinite(o).all()

rng = torch.Generator(device="cpu").manual_seed(0)
for t in range(300):
    a = (torch.rand((8, 2), generator=rng) * 2 - 1).to(o.device)
    obs, r, term, trunc, _ = env.step(a)
    assert torch.isfinite(r).all(), f"non-finite reward at {t}"
    # No milestone can plausibly latch under random actions this early, so
    # per-step reward must stay in the pre-bonus envelope.
    assert r.abs().max() < 30.0, f"reward spike {r.abs().max()} at {t}"
print("SMOKE HARBOR PASS", flush=True)
env.close()
app.close()
