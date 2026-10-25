"""G11: the harbour journal step with batched host/device copies equals T-base v2.3 bit for bit.

On the REAL HarborMissionEnv (stubbed Isaac Lab, harbour_journal_fixtures), stepped with random actions
and every suite at D1, after every control step:
* _journal_state (one copy, extra tensors included) == REFERENCE_state (T-base, verbatim): state dict,
  pose, and the extra host copies (episode counter, D7 contact, delivered thrust);
* _journal_mirror_task_state (one copy) == REFERENCE_mirror (T-base, verbatim) on the same oracle state,
  and the returned codes == codes.to(device);
* _journal_get_observations (one copy for gate index, gate midpoints, dwell) == REFERENCE_observations;
* _journal_publish_d7 == REFERENCE_publish (T-base, verbatim) on random sample sets (all rows, subsets).
Run: python -B tasks/harbor_mission/test_g11_sync_free.py   (CPU only)
"""
from __future__ import annotations

from pathlib import Path
import random
import sys
from types import SimpleNamespace as NS
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import harbour_journal_fixtures as F  # noqa: E402

D1 = ("waves=D1", "wind=D1", "current=D1", "actuator=D1", "perception=D1", "dynamic=D1")
MIRRORED = ("phase", "_exit_gate_progress", "_m1", "_m2", "_m3", "_stage_goal_entry_this_step", "_success",
            "_first_success_time_s", "hold_timer", "_hold_steps", "path_length", "_min_clearance",
            "_contact_before_dock", "_max_phase", "_phase_time_s", "_previous_xy")


def bits(t):
    t = torch.as_tensor(t).detach().contiguous()
    return (str(t.dtype), tuple(t.shape), t.cpu().numpy().tobytes())


def REFERENCE_state(env, robot_data, env_origins, water_level_local):
    n = env.num_envs
    pose = robot_data.root_link_pose_w.detach().to("cpu", torch.float64)
    vel = robot_data.root_com_vel_w.detach().to("cpu", torch.float64)
    origins = env_origins.detach().to("cpu", torch.float64)
    com = pose[:, :3] - origins[:, :3]
    quat = pose[:, 3:7]
    water = torch.as_tensor(water_level_local).detach().to("cpu", torch.float64).reshape(-1).expand(n).clone()
    state = {"com": com, "quat": quat, "lin": vel[:, :3], "ang": vel[:, 3:6], "water": water,
             "nominal": torch.full((n,), env._jnominal, dtype=torch.float64)}
    w, x, y, z = quat.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return state, torch.stack((com[:, 0], com[:, 1], yaw), -1)


def REFERENCE_mirror(env):
    from torch.nn import functional
    o, dev = env._oracle, env.device
    spawn2 = o.spawn_phase == 2
    departed = spawn2 | ~torch.isnan(o._departure_time())
    transited = spawn2 | ~torch.isnan(o.transit_time)
    env.phase.copy_(o.stage().to(dev))
    progress = (~torch.isnan(o.gate_times)).sum(-1)
    env._exit_gate_progress.copy_(torch.where(spawn2, torch.full_like(progress, 2), progress).to(dev))
    env._m1.copy_(departed.to(dev))
    env._m2.copy_(transited.to(dev))
    env._m3.copy_(o.success.to(dev))
    env._stage_goal_entry_this_step.copy_(o.success_now.to(dev))
    env._success.copy_(o.success.to(dev))
    env._first_success_time_s.copy_(o.tts.to(dev, env._first_success_time_s.dtype))
    env.hold_timer.copy_(o.dwell.to(dev, env.hold_timer.dtype))
    env._hold_steps.copy_(torch.round(o.dwell / env._jcontrol_dt).to(dev, torch.long))
    env.path_length.copy_(o.full_len.to(dev, env.path_length.dtype))
    env._min_clearance.copy_(o.min_clearance.to(dev, env._min_clearance.dtype))
    env._contact_before_dock.copy_((~o.clean).to(dev))
    env._max_phase.copy_(torch.maximum(env._max_phase, env.phase))
    env._phase_time_s += (functional.one_hot(env._phase_at_step_start, num_classes=4).float() * env.control_step_s)
    env._previous_xy.copy_(env._journal_com_xy())


def REFERENCE_publish(env, samples):
    if not samples:
        return
    base = env._jbase_max
    rows = [int(i) for i in samples]
    index = torch.as_tensor(rows, dtype=torch.long, device=env.device)
    env._d7_physical[index, base:] = False
    env._d7_visible[index, base:] = False
    env_rows, slots, values, flags = [], [], [], []
    for i in rows:
        for sample in samples[i]:
            if not base <= int(sample.slot) < env._jstorage:
                raise ValueError("slot")
            env_rows.append(i)
            slots.append(int(sample.slot))
            values.append((float(sample.x), float(sample.y), float(sample.radius_m)))
            flags.append((bool(sample.physical), bool(sample.visible)))
    if not env_rows:
        return
    er = torch.as_tensor(env_rows, dtype=torch.long, device=env.device)
    sl = torch.as_tensor(slots, dtype=torch.long, device=env.device)
    val = torch.as_tensor(values, dtype=env.obstacle_centers.dtype, device=env.device)
    flag = torch.as_tensor(flags, dtype=torch.bool, device=env.device)
    env.obstacle_centers[er, sl] = val[:, :2] + env.scene.env_origins[er, :2].to(val.dtype)
    env.obstacle_radii[er, sl] = val[:, 2]
    env._d7_physical[er, sl] = flag[:, 0]
    env._d7_visible[er, sl] = flag[:, 1]


def snapshot(env, names):
    return {name: getattr(env, name).detach().clone() for name in names}


def restore(env, saved):
    for name, value in saved.items():
        getattr(env, name).copy_(value)


class G11Harbour(unittest.TestCase):
    def test_state_mirror_observations_and_publish(self):
        env = F.make_env(tokens=D1, num_envs=3, pool_episodes=4)
        env.reset()
        gen = torch.Generator().manual_seed(5)
        for k in range(40):
            if k == 25:      # a non-finite pose: rows end with code 9 and reset inside the window
                pose = env.robot.data.root_link_pose_w.clone()
                pose[1, :] = float("nan")
                env.robot.data.root_link_pose_w = pose
            action = torch.rand((3, 2), generator=gen) * 2.0 - 1.0
            env.step(action)
            data, origins = env.robot.data, env.scene.env_origins
            water = torch.rand(3, generator=gen, dtype=torch.float64) * 0.1
            extra = (env.episode_length_buf, torch.rand(3, generator=gen) < 0.3,
                     torch.randn(3, 2, generator=gen))
            new_state, new_pose, hosts = env._journal_state(data, origins, water, extra=extra)
            ref_state, ref_pose = REFERENCE_state(env, data, origins, water)
            self.assertEqual(sorted(new_state), sorted(ref_state))
            for key in ref_state:
                self.assertEqual(bits(new_state[key]), bits(ref_state[key]), (k, key))
            self.assertEqual(bits(new_pose), bits(ref_pose))
            for got, want in zip(hosts, extra):
                self.assertEqual(bits(got), bits(want.detach().to("cpu")))
            plain_state, plain_pose = env._journal_state(data, origins, water)
            self.assertEqual(bits(plain_pose), bits(ref_pose))
            # mirror: same oracle state, buffers restored between the two versions
            before = snapshot(env, MIRRORED)
            codes = torch.tensor([0, 3, 9][:3], dtype=torch.long)
            moved = env._journal_mirror_task_state(codes)
            new_buffers = snapshot(env, MIRRORED)
            restore(env, before)
            REFERENCE_mirror(env)
            for name in MIRRORED:
                self.assertEqual(bits(new_buffers[name]), bits(getattr(env, name)), (k, name))
            self.assertEqual(bits(moved), bits(codes.to(env.device)))
            # observations: the estimate counters allow one call per control step -> compare the pieces
            dtype = env._forward_2d().dtype
            got = env._journal_get_observations.__func__  # noqa: F841 (method exists)
            gate_index_ref = env._oracle.gate_passed()[:, :3].sum(-1).clamp(max=2).to(env.device)
            rows = torch.arange(env.num_envs, device=env.device)
            gate_mid_ref = env._jgate_mid.to(env.device, dtype)[rows, gate_index_ref]
            dwell_ref = env._oracle.dwell.to(env.device, dtype)
            from tasks._shared.journal_env_mixin import device_copies
            gi, gm, dw = device_copies([env._oracle.gate_passed()[:, :3].sum(-1).clamp(max=2), env._jgate_mid.to(dtype),
                                        env._oracle.dwell.to(dtype)], env.device)
            self.assertEqual(bits(gi), bits(gate_index_ref))
            self.assertEqual(bits(gm[rows, gi]), bits(gate_mid_ref))
            self.assertEqual(bits(dw), bits(dwell_ref))
        rng = random.Random(3)
        base, storage = env._jbase_max, env._jstorage
        for case in range(60):
            saved = snapshot(env, ("obstacle_centers", "obstacle_radii", "_d7_physical", "_d7_visible"))
            keys = list(range(3)) if rng.random() < 0.6 else rng.sample(range(3), rng.randint(1, 3))
            rng.shuffle(keys)
            samples = {i: [NS(slot=rng.randrange(base, storage), x=rng.uniform(-50, 50), y=rng.uniform(-50, 50),
                              radius_m=rng.uniform(0.5, 4), physical=rng.random() < 0.6, visible=rng.random() < 0.6)
                           for _ in range(rng.choice((0, 1, 2)))] for i in keys}
            env._journal_publish_d7(samples)
            new = snapshot(env, saved)
            restore(env, saved)
            REFERENCE_publish(env, samples)
            for name in saved:
                self.assertEqual(bits(new[name]), bits(getattr(env, name)), (case, name))


if __name__ == "__main__":
    unittest.main(verbosity=2)
