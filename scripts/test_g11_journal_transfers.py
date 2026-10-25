"""G11: batched host/device copies of the journal step path equal the per-tensor copies bit for bit.

* journal_env_mixin.host_copies / device_copies: dtype, shape and raw bytes of every result equal
  t.detach().to("cpu") / t.to(device) (float32 / float64 / float16 / int64 / int32 / bool / uint8,
  NaN payloads, signed zeros, infinities, non-contiguous views, 0-dim and empty tensors, mixed devices).
* JournalEnvMixin._journal_poses / _journal_d7_step: the T-base v2.3 per-quantity implementation
  (REFERENCE_* below, verbatim) and the batched one give identical Pose rows and ticks on random robot
  states with non-finite rows (the D18 rule: non-finite rows are left out).
Run: python -B -m pytest -p no:cacheprovider scripts/test_g11_journal_transfers.py -q
"""
from __future__ import annotations

import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tasks._shared import journal_env_mixin as M  # noqa: E402


def raw(t):
    t = t.detach().contiguous()
    return (str(t.dtype), tuple(t.shape), t.cpu().numpy().tobytes())


def nan_with_payload(dtype):
    if dtype == torch.float32:
        return torch.tensor([struct.unpack("<f", struct.pack("<I", 0x7FC12345))[0]], dtype=dtype)
    return torch.tensor([struct.unpack("<d", struct.pack("<Q", 0x7FF8000000ABCDEF))[0]], dtype=dtype)


def samples():
    g = torch.Generator().manual_seed(11)
    out = []
    for dtype in (torch.float32, torch.float64, torch.float16):
        base = torch.randn(7, 5, generator=g).to(dtype)
        base[0, 0] = -0.0
        base[1, 1] = math.inf
        base[2, 2] = -math.inf
        if dtype != torch.float16:
            base[3, 3] = nan_with_payload(dtype)[0]
        out += [base, base[:, 1:4], base.t(), base[2], base[0, 0]]
    out += [torch.arange(9, dtype=torch.int64).reshape(3, 3), torch.tensor([2**62, -5]),
            torch.tensor([True, False, True]), torch.tensor([1, 255], dtype=torch.uint8),
            torch.tensor([7, -7], dtype=torch.int32), torch.zeros((0, 3)), torch.zeros((0,), dtype=torch.bool),
            torch.tensor(3.5).expand(1), torch.tensor(-2.0, dtype=torch.float64).expand(4), torch.tensor(True).expand(3, 2)]
    return out


def test_host_copies_equal_per_tensor_copies():
    tensors = samples()
    got = M.host_copies(tensors)
    assert [raw(a) for a in got] == [raw(t.detach().to("cpu")) for t in tensors]
    assert all(a.device.type == "cpu" for a in got)
    for order in (list(reversed(tensors)), tensors[::3], tensors[:1], []):
        assert [raw(a) for a in M.host_copies(order)] == [raw(t.to("cpu")) for t in order]


def test_device_copies_equal_per_tensor_copies():
    tensors = samples()
    got = M.device_copies(tensors, "cpu")
    assert [raw(a) for a in got] == [raw(t.to("cpu")) for t in tensors]
    # results are fresh memory (never the caller's storage), like a copy to another device
    assert all(a.data_ptr() != t.data_ptr() or t.numel() == 0 for a, t in zip(got, tensors))
    assert M.device_copies([], "cpu") == []


# --------------------------------------------- T-base v2.3 reference poses --
def REFERENCE_poses(robot_data, env_origins):
    pose = robot_data.root_link_pose_w
    vel = robot_data.root_com_vel_w
    xy = (pose[:, :2] - env_origins[:, :2]).detach().to("cpu", torch.float64)
    yaw = M._yaw_from_quat(pose[:, 3:7].detach().to("cpu", torch.float64))
    v = vel[:, :2].detach().to("cpu", torch.float64)
    surge = v[:, 0] * torch.cos(yaw) + v[:, 1] * torch.sin(yaw)
    Pose = M._dynamic().Pose
    rows = torch.stack((xy[:, 0], xy[:, 1], yaw, surge), -1).tolist()
    return {i: Pose(float(r[0]), float(r[1]), float(r[2]), float(r[3])) for i, r in enumerate(rows)
            if all(math.isfinite(v) for v in r)}


def robot_states(seed, n=9):
    g = torch.Generator().manual_seed(seed)
    pose = torch.randn(n, 7, generator=g) * 40.0
    pose[:, 3:7] = torch.nn.functional.normalize(pose[:, 3:7], dim=-1)
    vel = torch.randn(n, 6, generator=g) * 3.0
    origins = torch.randn(n, 3, generator=g) * 100.0
    if seed % 2:
        pose[1, 0] = math.nan
        pose[2, 4] = math.inf          # infinite quaternion component: yaw may stay finite
        vel[3, 1] = -math.inf
        pose[4, 3:7] = math.nan
    return NS(root_link_pose_w=pose, root_com_vel_w=vel), origins


def pose_bits(poses):
    return {i: tuple(struct.pack("<d", float(getattr(p, f))) for f in ("x", "y", "yaw", "speed"))
            if hasattr(p, "speed") else repr(p) for i, p in poses.items()}


class FakeMixinEnv(M.JournalEnvMixin):
    def __init__(self, n):
        self.num_envs = n
        self._d7_tick = torch.arange(n, dtype=torch.long) * 7 + 3
        self.samples_seen = []

        class Bridge:
            def step(inner, ticks, poses):
                self.samples_seen.append((dict(ticks), pose_bits(poses)))
                return {}
        self._d7_bridge = Bridge()


@pytest.mark.parametrize("seed", range(6))
def test_journal_poses_equal_the_per_quantity_reference(seed):
    data, origins = robot_states(seed)
    env = FakeMixinEnv(9)
    assert pose_bits(env._journal_poses(data, origins)) == pose_bits(REFERENCE_poses(data, origins))


@pytest.mark.parametrize("seed", range(4))
def test_d7_step_hands_the_bridge_the_reference_ticks_and_poses(seed):
    data, origins = robot_states(seed)
    env = FakeMixinEnv(9)
    env._journal_d7_step(data, origins)
    ticks = [int(t) for t in env._d7_tick.detach().to("cpu").tolist()]
    poses = REFERENCE_poses(data, origins)
    expected = ({i: t for i, t in enumerate(ticks) if i in poses}, pose_bits(poses))
    assert env.samples_seen == [expected]


def test_poses_need_one_host_transfer_per_call():
    """The batched path moves everything with ONE .to('cpu') of one concatenated buffer."""
    data, origins = robot_states(0)
    env = FakeMixinEnv(9)
    calls = []

    class Probe(torch.overrides.TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            if getattr(func, "__name__", "") in ("to", "cpu") and args and isinstance(args[0], torch.Tensor):
                target = args[1] if len(args) > 1 else (kwargs or {}).get("device")
                if target == "cpu" or getattr(target, "type", None) == "cpu" or func.__name__ == "cpu":
                    calls.append(tuple(args[0].shape))
            return func(*args, **(kwargs or {}))

    with Probe():
        env._journal_d7_step(data, origins)
    # one blob transfer (uint8, concatenated), then host-side widening of the three float parts
    blob = [c for c in calls if len(c) == 1 and c[0] == 9 * 8 + 9 * 2 * 4 + 9 * 4 * 4 + 9 * 2 * 4]
    assert len(blob) == 1, calls
