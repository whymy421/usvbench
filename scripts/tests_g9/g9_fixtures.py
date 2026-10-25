"""Shared CPU fixtures of the G9 tests (CUDA-graph MPPI, scripts/mppi_controller.CudaGraphMPPI).

* ReplayBackend: the CPU stand-in for mppi_controller.CudaGraphBackend. capture() runs the
  step function once and its outputs become the STATIC outputs; replay() re-runs the
  function (which reads only the wrapper's static buffers) and copies the fresh results
  into those static outputs. Everything of the graph implementation except the CUDA capture
  itself runs: static input buffers, the static key buffer, the static warm start, sigma
  hoisting, recapture, release, fail-closed paths. Optional op tracing of each replay.
* OpTrace: a TorchDispatchMode that records every aten op (name, tensor shapes / dtypes,
  scalar arguments) in execution order.
* make_inputs(mode, ...): deterministic per-step rollout inputs for the five modes, journal
  style (36-ray obstacle discs from journal_controllers.ray_obstacles, scaled by the vessel
  length ratio).
* frozen(vessel): the vessel model and its FROZEN journal MPPI parameters
  (mppi_controller.journal_mppi_params: BlueBoat K 256 / H 60, WAM-V K 256 / H 269).
No Isaac, no CUDA (CUDA is hidden before torch is imported).
"""
from __future__ import annotations

import dataclasses
import math
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

import journal_controllers as jc  # noqa: E402
import mppi_controller as mc  # noqa: E402

torch.set_num_threads(1)

MODES = ("goal", "gates", "hold", "dock", "mission")
VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")


def _norm(value):
    if isinstance(value, torch.Tensor):
        return ("T", tuple(value.shape), str(value.dtype))
    if isinstance(value, (list, tuple)):
        return tuple(_norm(v) for v in value)
    if isinstance(value, float):
        return ("f", value.hex())
    return repr(value)


class OpTrace(TorchDispatchMode):
    """Every aten op in execution order: (op, normalized args, normalized kwargs)."""

    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        self.ops.append((str(func), _norm(args), tuple(sorted((k, _norm(v)) for k, v in kwargs.items()))))
        return out


class ReplayBackend:
    """CPU stand-in for CudaGraphBackend (see the module docstring)."""

    name = "cpu_replay_stand_in"
    capture_error_mode = "n/a"

    def __init__(self, fail_warmup=None, fail_capture=None, trace=False):
        self.fail_warmup = fail_warmup
        self.fail_capture = fail_capture
        self.trace = trace
        self.last_trace = None
        self.events = []
        self.live = 0
        self._next = 0

    def warmup(self, fn, iterations):
        self.events.append(("warmup", int(iterations)))
        if self.fail_warmup is not None:
            raise self.fail_warmup
        for _ in range(int(iterations)):
            fn()

    def capture(self, fn):
        self.events.append(("capture",))
        if self.fail_capture is not None:
            raise self.fail_capture
        outputs = fn()
        self._next += 1
        self.live += 1
        return {"fn": fn, "outputs": outputs, "id": self._next}, outputs

    def replay(self, handle):
        self.events.append(("replay", handle["id"]))
        if self.trace:
            with OpTrace() as trace:
                fresh = handle["fn"]()
            self.last_trace = trace.ops
        else:
            fresh = handle["fn"]()
        for static, new in zip(handle["outputs"], fresh):
            static.copy_(new)

    def release(self, handle):
        self.events.append(("release", handle["id"]))
        self.live -= 1


def frozen(vessel, seed=3, K=None, H=None):
    """(dynamics, params) with the FROZEN journal parameters of ``vessel`` (optionally K/H reduced)."""
    dynamics = mc.VesselDynamicsV2(vessel)
    params, _stamp = mc.journal_mppi_params(dynamics.vessel, mc.MPPIParams(seed=seed))
    if K is not None or H is not None:
        params = dataclasses.replace(params, K=K or params.K, H=H or params.H)
    return dynamics, params


def controllers(n, params, dynamics, count=2):
    return [mc.MPPIController(num_envs=n, params=params, dynamics=dynamics) for _ in range(count)]


def make_inputs(mode, n, step, scale=1.0, rays=36, gates=4, seed=0):
    """Deterministic rollout inputs of one control step (keyword dict incl. ``state``)."""
    g = torch.Generator().manual_seed(7919 * (seed + 1) + 31 * step)
    pos = torch.randn((n, 2), generator=g) * 4.0 * scale
    yaw = (torch.rand((n,), generator=g) * 2.0 - 1.0) * math.pi
    vel = torch.randn((n, 3), generator=g) * torch.tensor([0.6, 0.1, 0.3])
    state = torch.stack((pos[:, 0], pos[:, 1], yaw, vel[:, 0], vel[:, 1], vel[:, 2]), dim=-1)
    ranges = torch.full((n, rays), 30.0 * scale)
    ranges[:, : rays // 4] = (torch.rand((n, rays // 4), generator=g) * 10.0 + 1.5) * scale
    centers, radii, active = jc.ray_obstacles(pos, yaw, ranges, 30.0 * scale)
    heading = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
    inputs = {"state": state, "mode": mode, "obstacle_centers": centers, "obstacle_radii": radii,
              "obstacle_active": active}
    if mode == "goal":
        inputs.update(targets=(pos + 20.0 * scale * heading.roll(1, 0))[:, None, :], goal_radius=2.0 * scale)
    elif mode == "gates":
        chain = pos[:, None, :] + (torch.arange(1, gates + 1, dtype=torch.float32)[None, :, None]
                                   * 6.0 * scale * heading[:, None, :])
        inputs.update(targets=chain, goal_radius=2.0 * scale,
                      active_index=torch.full((n,), min(step // 3, gates - 1), dtype=torch.long))
    elif mode == "hold":
        inputs.update(targets=(pos + torch.randn((n, 2), generator=g) * 1.5 * scale)[:, None, :],
                      goal_radius=2.0 * scale)
    elif mode == "dock":
        dock = torch.randn((n, 2), generator=g)
        inputs.update(targets=(pos + 3.0 * scale * heading)[:, None, :], goal_radius=2.5 * scale,
                      dock_heading=dock / dock.norm(dim=-1, keepdim=True))
    elif mode == "mission":
        gate = pos + 2.0 * scale * heading
        berth = pos + 25.0 * scale * heading
        inputs.update(targets=torch.stack((gate, berth), dim=1), goal_radius=2.5 * scale,
                      active_index=torch.full((n,), 1 if step >= 4 else 0, dtype=torch.long),
                      dock_heading=heading.clone(), gate_normals=heading[:, None, :].clone(),
                      gate_half_width_m=3.0 * scale, gate_min_normal_speed_mps=0.2)
    else:
        raise ValueError(mode)
    return inputs


def call(controller, inputs):
    kwargs = dict(inputs)
    state = kwargs.pop("state")
    return controller.rollout(state, **kwargs)


def assert_step_equal(eager, graph_wrapper, action_e, info_e, action_g, info_g, expected_keys, where=""):
    """Bitwise: action, every info tensor (same keys, same order), warm start, keys, counters."""
    assert torch.equal(action_e, action_g), "action differs %s: max %r" % (
        where, float((action_e.double() - action_g.double()).abs().max()))
    assert list(info_e) == list(info_g), where
    for key in info_e:
        assert torch.equal(info_e[key], info_g[key]), "info %s differs %s" % (key, where)
    assert torch.equal(eager.U, graph_wrapper.U), "warm start differs %s" % where
    assert graph_wrapper.last_keys == expected_keys, "noise keys differ %s" % where
    assert graph_wrapper.controller.control_steps == eager.control_steps, where
    assert graph_wrapper.controller.episode_ids == eager.episode_ids, where


def run_side_by_side(eager, wrapper, mode, steps, n, scale=1.0, resets=(), seed=0, inputs_for=None):
    """Eager and graph on the same inputs every step, bitwise equal; resets as (step, mask, episode_ids)."""
    inputs_for = inputs_for or (lambda step: make_inputs(mode, n, step, scale=scale, seed=seed))
    for step in range(steps):
        inputs = inputs_for(step)
        expected = eager.noise_keys()
        action_e, info_e = call(eager, inputs)
        action_g, info_g = call(wrapper, inputs)
        assert_step_equal(eager, wrapper, action_e, info_e, action_g, info_g, expected,
                          where="(mode %s step %d)" % (mode, step))
        for when, mask, episode_ids in resets:
            if when == step:
                done = torch.tensor(mask, dtype=torch.bool)
                eager.reset(done, episode_ids)
                wrapper.reset(done, episode_ids)
    return wrapper
