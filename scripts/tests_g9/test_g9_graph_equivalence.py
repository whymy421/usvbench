"""G9: the CUDA-graph MPPI implementation executes the eager arithmetic, in the eager order.

CPU, with the replay stand-in backend (g9_fixtures.ReplayBackend: the graph's static buffers,
key buffer, warm start and outputs are real; only the CUDA capture is replaced by re-running
the captured function):

* bitwise equality of actions, every info tensor, warm starts, noise keys and counters, step
  by step, in all five modes (goal, gates, hold, dock, mission), with resets (implicit and
  explicit episode ids) and the warm-start shift, at both vessels' FROZEN journal parameters
  (BlueBoat K 256 / H 60, WAM-V K 256 / H 269, mppi_params_v2.json) and on the legacy
  BlueBoat model (circle clearance);
* the op trace of the captured step equals the eager rollout's op trace, op for op (names,
  shapes, dtypes, scalar constants), except the three host constants the graph hoists: the
  sigma tuple and the noise-key list copies (aten.lift_fresh) and the key tensor's detach_;
* the graph's static noise equals MPPIController.sample_noise bit for bit.
Run: python -B -m pytest -p no:cacheprovider scripts/tests_g9/test_g9_graph_equivalence.py -q
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402

import g9_fixtures as fx  # noqa: E402
from g9_fixtures import mc, torch  # noqa: E402

# Resets: env 1 after step 2 (episode id advanced implicitly), env 0 after step 4 with an
# explicit episode id, both after step 6.
RESETS = ((2, [False, True], None), (4, [True, False], [9, 1]), (6, [True, True], None))


def _scale(vessel):
    return mc.vessel_length_scale(vessel)


@pytest.mark.parametrize("mode", fx.MODES)
def test_blueboat_frozen_parameters_bitwise_in_every_mode_with_resets(mode):
    dynamics, params = fx.frozen("blueboat_v2")
    assert (params.K, params.H, params.lambda_) == (256, 60, 0.3)
    eager, inner = fx.controllers(2, params, dynamics)
    graph = mc.CudaGraphMPPI(inner, backend=fx.ReplayBackend())
    fx.run_side_by_side(eager, graph, mode, 8, 2, scale=_scale("blueboat_v2"), resets=RESETS)
    assert eager.episode_ids == inner.episode_ids == [10, 2]
    assert graph.implementation_stamp() == {"mode": "cuda_graph_v1", "captured_graphs": 1, "recaptures": 0}


@pytest.mark.parametrize("mode", fx.MODES)
def test_wamv_frozen_parameters_bitwise_in_every_mode_with_resets(mode):
    dynamics, params = fx.frozen("wamv_vrx_v2_opt1")
    assert (params.K, params.H) == (256, 269)
    eager, inner = fx.controllers(2, params, dynamics)
    graph = mc.CudaGraphMPPI(inner, backend=fx.ReplayBackend())
    fx.run_side_by_side(eager, graph, mode, 5, 2, scale=_scale("wamv_vrx_v2_opt1"),
                        resets=((1, [False, True], None), (3, [True, False], [4, 1])))
    assert graph.captured_graphs == 1 and graph.recaptures == 0


@pytest.mark.parametrize("mode", fx.MODES)
def test_legacy_blueboat_model_bitwise(mode):
    """The legacy NominalBlueBoatDynamics (circle clearance, half beam) under the graph too."""
    params = mc.MPPIParams(seed=11, K=64)
    dynamics = mc.NominalBlueBoatDynamics()
    eager, inner = fx.controllers(3, params, dynamics)
    graph = mc.CudaGraphMPPI(inner, backend=fx.ReplayBackend())
    fx.run_side_by_side(eager, graph, mode, 6, 3, resets=((2, [False, True, False], None),))


def test_more_envs_than_two_and_a_different_controller_seed():
    dynamics, params = fx.frozen("blueboat_v2", seed=21001)
    eager, inner = fx.controllers(5, params, dynamics)
    graph = mc.CudaGraphMPPI(inner, backend=fx.ReplayBackend())
    fx.run_side_by_side(eager, graph, "goal", 4, 5, resets=((1, [True, False, False, True, False], None),))


def test_warm_start_shift_is_the_eager_shift():
    """After a step, U = (u_new[1:], u_new[-1]) in both; the last two rows are equal (repeat)."""
    dynamics, params = fx.frozen("blueboat_v2", K=64)
    eager, inner = fx.controllers(2, params, dynamics)
    graph = mc.CudaGraphMPPI(inner, backend=fx.ReplayBackend())
    fx.run_side_by_side(eager, graph, "goal", 3, 2)
    assert torch.equal(inner.U[:, -1, :], inner.U[:, -2, :])
    assert inner.U is graph._live.u_static           # the controller's U IS the static warm start


def test_static_noise_equals_sample_noise():
    dynamics, params = fx.frozen("blueboat_v2", K=16, H=7)
    controller = mc.MPPIController(num_envs=3, params=params, dynamics=dynamics)
    controller.episode_ids = [0, 4, 2]
    controller.control_steps = [5, 0, 11]
    keys = torch.tensor(controller.noise_keys(), dtype=torch.int64)
    assert torch.equal(controller.keyed_normals(keys, 16, 7), controller.sample_noise(16, 7))
    assert controller.noise_keys() == [mc.noise_key(params.seed, e, ep, st) for e, ep, st in
                                       zip(controller.env_ids, controller.episode_ids, controller.control_steps)]


def _host_constant(op):
    return op[0] in ("aten.lift_fresh.default", "aten.detach_.default")


@pytest.mark.parametrize("vessel,mode", [("blueboat_v2", m) for m in fx.MODES] + [("wamv_vrx_v2_opt1", "mission")])
def test_captured_step_runs_the_eager_ops_in_the_eager_order(vessel, mode):
    """Op trace of a replayed step == op trace of the eager rollout minus the hoisted host constants."""
    dynamics, params = fx.frozen(vessel)
    n, scale = 2, _scale(vessel)
    eager, inner = fx.controllers(n, params, dynamics)
    backend = fx.ReplayBackend(trace=True)
    graph = mc.CudaGraphMPPI(inner, backend=backend)
    for step in range(3):
        inputs = fx.make_inputs(mode, n, step, scale=scale)
        with torch.inference_mode():
            with fx.OpTrace() as trace:
                action_e, _ = fx.call(eager, inputs)
        action_g, _ = fx.call(graph, inputs)
        assert torch.equal(action_e, action_g)
        if step == 0:
            # the first eager call of a fresh dynamics object also fills the mixer's
            # prepared-parameter cache (validation ops, host syncs); later steps do not
            continue
        eager_ops, graph_ops = trace.ops, backend.last_trace
        hoisted = [op for op in eager_ops if _host_constant(op)]
        # sigma (2,) float32 and the keys (n,) int64 lifted from host data; the keys' detach_
        assert [(op[0], op[1][0][1:]) for op in hoisted] == [
            ("aten.lift_fresh.default", ((2,), "torch.float32")),
            ("aten.lift_fresh.default", ((n,), "torch.int64")),
            ("aten.detach_.default", ((n,), "torch.int64"))], hoisted
        assert [op for op in eager_ops if not _host_constant(op)] == graph_ops
        assert not any(_host_constant(op) for op in graph_ops)
