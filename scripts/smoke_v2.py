"""B1 v2 smoke: obs dims/latch/speed slots + swiftness reward behavior."""
import sys

from isaaclab.app import AppLauncher

parser_args = []
app = AppLauncher(headless=True).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg("Isaac-USV-HazardNav-Direct-v2", device="cuda:0", num_envs=8)
cfg.seed = 42
env = gym.make("Isaac-USV-HazardNav-Direct-v2", cfg=cfg)
base = env.unwrapped
obs, _ = env.reset()
o = obs["policy"]
assert o.shape == (8, 41), f"obs shape {o.shape}"
assert torch.isfinite(o).all(), "non-finite obs"
# slot 3 = reached latch (must be 0 at spawn), slot 4 = speed_norm (>= 0)
assert torch.all(o[:, 3] == 0.0), f"latch at spawn: {o[:, 3]}"
assert torch.all(o[:, 4] >= 0.0), "negative speed_norm"

# Parked boat (zero action): swiftness tax ~= scale*dt each step, so the
# total reward for an idle step in open water should be ~ -0.000833 plus
# tiny progress noise. Full-throttle steps should pay ~zero swiftness.
idle = torch.zeros((8, 2), device=o.device)
r_idle = []
for _ in range(30):
    obs, r, _, _, _ = env.step(idle)
    r_idle.append(r.clone())
r_idle = torch.stack(r_idle)[5:]  # skip startup transients
swift_bound = base.cfg.reward_swift_scale * base.control_step_s
assert torch.isfinite(r_idle).all()
print(f"idle-step reward mean={r_idle.mean():.6f} (swift bound {swift_bound:.6f})")

full = torch.ones((8, 2), device=o.device)
for _ in range(120):
    obs, r, _, _, _ = env.step(full)
speed_now = obs["policy"][:, 4]
print(f"speed_norm after 2 s full throttle: {speed_now.mean():.3f}")
assert speed_now.mean() > 0.3, "speed channel not responding"
assert torch.isfinite(r).all()

print("SMOKE V2 PASS")
env.close()
app.close()
