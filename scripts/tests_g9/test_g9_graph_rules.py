"""G9: recapture rules, stale-input protection and the fail-closed paths of CudaGraphMPPI.

CPU, replay stand-in backend (g9_fixtures.ReplayBackend). Every scenario also checks that the
graph stays bitwise equal to an eager controller driven with the same inputs.

* one signature -> one capture; every change of a tensor shape, dtype or presence, of a scalar
  argument, of the mode, of an MPPIParams value, of the dynamics object (vessel) or of the
  warm-start horizon recaptures, releases the previous graph first (one live graph) and is
  counted (captured_graphs, recaptures) with its reason;
* inputs are restaged every step (in-place changes of the caller's tensors are seen, a
  warm start rebound outside the wrapper is copied in), the mixer's prepared parameters are
  pinned with the graph;
* fail closed: a warmup or capture failure raises GraphCaptureError (no eager fallback, the
  controller state untouched); the eager rollout's own input validation raises the same
  ValueError; graph mode refuses the legacy noise stream, non-CUDA devices without a stand-in
  backend, tensor scalars and non-tensor inputs.
Run: python -B -m pytest -p no:cacheprovider scripts/tests_g9/test_g9_graph_rules.py -q
"""
from __future__ import annotations

import dataclasses
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402

import g9_fixtures as fx  # noqa: E402
from g9_fixtures import mc, torch  # noqa: E402

N = 2


def _pair(K=48, H=None, vessel="blueboat_v2", backend=None, n=N, seed=5):
    dynamics, params = fx.frozen(vessel, seed=seed, K=K, H=H)
    eager, inner = fx.controllers(n, params, dynamics)
    backend = backend or fx.ReplayBackend()
    return eager, inner, mc.CudaGraphMPPI(inner, backend=backend), backend


def _step(eager, graph, inputs, where=""):
    expected = eager.noise_keys()
    action_e, info_e = fx.call(eager, inputs)
    action_g, info_g = fx.call(graph, inputs)
    fx.assert_step_equal(eager, graph, action_e, info_e, action_g, info_g, expected, where)


def test_one_signature_one_capture_and_no_release():
    eager, _, graph, backend = _pair()
    for step in range(5):
        _step(eager, graph, fx.make_inputs("goal", N, step), step)
    assert graph.implementation_stamp() == {"mode": "cuda_graph_v1", "captured_graphs": 1, "recaptures": 0}
    assert [e[0] for e in backend.events].count("capture") == 1
    assert backend.events[0] == ("warmup", mc.GRAPH_WARMUP_ITERATIONS) == ("warmup", 3)
    assert not any(e[0] == "release" for e in backend.events) and backend.live == 1
    assert graph.captures == [{"index": 1, "reason": ["first capture"], "seconds": graph.captures[0]["seconds"]}]


def _change(name):
    """Input transformations, one signature part each (applied from step 2 on)."""
    def fewer_rays(inputs):
        out = dict(inputs)
        for key in ("obstacle_centers", "obstacle_radii", "obstacle_active"):
            out[key] = out[key][:, :24].contiguous()
        return out

    def no_obstacles(inputs):
        out = dict(inputs)
        for key in ("obstacle_centers", "obstacle_radii", "obstacle_active"):
            out[key] = None
        return out

    def no_active_mask(inputs):
        return dict(inputs, obstacle_active=None)

    def float64_state(inputs):
        return dict(inputs, state=inputs["state"].double())

    def radius(inputs):
        return dict(inputs, goal_radius=inputs["goal_radius"] + 0.5)

    def hold(inputs):
        return dict(inputs, mode="hold")

    return {"obstacle_centers": fewer_rays, "obstacle_presence": no_obstacles, "obstacle_active": no_active_mask,
            "state_dtype": float64_state, "goal_radius": radius, "mode": hold}[name]


@pytest.mark.parametrize("name,reason", [("obstacle_centers", "obstacle_centers"),
                                         ("obstacle_presence", "obstacle_centers"),
                                         ("obstacle_active", "obstacle_active"),
                                         ("state_dtype", "state"),
                                         ("goal_radius", "goal_radius"),
                                         ("mode", "mode")])
def test_input_signature_change_recaptures_and_stays_equal(name, reason):
    eager, _, graph, backend = _pair()
    change = _change(name)
    for step in range(5):
        inputs = fx.make_inputs("goal", N, step)
        _step(eager, graph, change(inputs) if step >= 2 else inputs, (name, step))
    assert graph.implementation_stamp() == {"mode": "cuda_graph_v1", "captured_graphs": 2, "recaptures": 1}
    assert graph.captures[1]["reason"] and graph.captures[1]["reason"][0].startswith(reason), graph.captures
    releases = [e for e in backend.events if e[0] == "release"]
    assert releases == [("release", 1)] and backend.live == 1              # the old graph released first
    assert backend.events.index(("release", 1)) < [i for i, e in enumerate(backend.events) if e == ("capture",)][1]


def test_presence_change_of_active_index_and_back_recaptures_each_time():
    eager, _, graph, _ = _pair()
    for step in range(6):
        inputs = fx.make_inputs("gates", N, step)
        if step in (2, 3):
            inputs["active_index"] = None
        _step(eager, graph, inputs, step)
    # gates (index) -> no index -> index: two signature changes
    assert graph.implementation_stamp() == {"mode": "cuda_graph_v1", "captured_graphs": 3, "recaptures": 2}


def test_gate_count_change_recaptures():
    eager, _, graph, _ = _pair()
    for step in range(4):
        inputs = fx.make_inputs("gates", N, step, gates=4 if step < 2 else 6)
        _step(eager, graph, inputs, step)
    assert graph.recaptures == 1 and "targets" in graph.captures[1]["reason"][0]


def test_parameter_change_recaptures():
    """MPPIParams values are baked into the graph: a changed value is a new signature."""
    eager, inner, graph, _ = _pair()
    for step in range(4):
        if step == 2:
            eager.params.w_obstacle = 55.0          # the shared params object: both controllers see it
        _step(eager, graph, fx.make_inputs("goal", N, step), step)
    assert graph.recaptures == 1 and graph.captures[1]["reason"] == ["params"]


def test_horizon_change_rebuilds_the_warm_start_like_eager_and_recaptures():
    eager, inner, graph, _ = _pair(H=20)
    for step in range(4):
        if step == 2:
            eager.params.H = 12                    # eager rebuilds U (zeros) when U has not H steps
        _step(eager, graph, fx.make_inputs("goal", N, step), step)
    assert inner.U.shape == (N, 12, 2) and graph.recaptures == 1
    assert "params" in graph.captures[1]["reason"] and "controller" in graph.captures[1]["reason"]


def test_vessel_change_recaptures():
    """A different dynamics object (another vessel) is a new signature, never a stale graph."""
    eager, inner, graph, _ = _pair()
    wamv = mc.VesselDynamicsV2("wamv_vrx_v2_opt1")
    for step in range(4):
        if step == 2:
            eager.dynamics = wamv
            inner.dynamics = wamv
        _step(eager, graph, fx.make_inputs("goal", N, step), step)
    assert graph.recaptures == 1 and graph.captures[1]["reason"] == ["dynamics"]


def test_inputs_are_restaged_every_step_even_when_changed_in_place():
    """The same caller tensors, mutated in place between steps, must reach the graph."""
    eager, _, graph, _ = _pair()
    inputs = fx.make_inputs("dock", N, 0)
    for step in range(4):
        fresh = fx.make_inputs("dock", N, step)
        for key, value in fresh.items():
            if isinstance(value, torch.Tensor):
                inputs[key].copy_(value)
        _step(eager, graph, inputs, step)
    assert graph.captured_graphs == 1


def test_warm_start_rebound_outside_the_wrapper_is_copied_in():
    """An eager call on the wrapped controller rebinds U; the next replay starts from its values."""
    eager, inner, graph, _ = _pair()
    reference = mc.MPPIController(num_envs=N, params=eager.params, dynamics=eager.dynamics)
    sequence = [fx.make_inputs("goal", N, step) for step in range(4)]
    fx.call(graph, sequence[0])
    fx.call(reference, sequence[0])
    fx.call(inner, sequence[1])                    # eager on the wrapped controller: U rebound
    fx.call(reference, sequence[1])
    static = graph._live.u_static
    assert inner.U is not static
    for step in (2, 3):
        _step(reference, graph, sequence[step], step)
    assert inner.U is static and graph.captured_graphs == 1


def test_prepared_mixer_parameters_are_pinned_with_the_graph():
    eager, inner, graph, _ = _pair()
    fx.call(graph, fx.make_inputs("goal", N, 0))
    prepared = dict(inner.dynamics._prepared)
    pinned = graph._live.pins[1]
    assert prepared and all(pinned[key] is value for key, value in prepared.items())
    inner.dynamics._prepared = {}                  # a later cache eviction cannot free them
    assert all(isinstance(v, mc._shared_module("thruster_mixer").ThrusterParameters) for v in pinned.values())


def test_capture_failure_fails_closed_without_eager_fallback():
    backend = fx.ReplayBackend(fail_capture=RuntimeError("operation not permitted when stream is capturing"))
    eager, inner, graph, _ = _pair(backend=backend)
    logged = []
    graph._log = logged.append
    calls = []
    original = mc.MPPIController.rollout

    def spy(self, *args, **kwargs):
        calls.append(self is inner)
        return original(self, *args, **kwargs)

    u_before, steps_before = inner.U.clone(), list(inner.control_steps)
    mc.MPPIController.rollout = spy
    try:
        with pytest.raises(mc.GraphCaptureError, match="capture failed .*operation not permitted.*never falls back"):
            fx.call(graph, fx.make_inputs("goal", N, 0))
    finally:
        mc.MPPIController.rollout = original
    assert not any(calls)                          # the wrapped controller's eager rollout never ran
    assert torch.equal(inner.U, u_before) and inner.control_steps == steps_before
    assert graph.implementation_stamp() == {"mode": "cuda_graph_v1", "captured_graphs": 0, "recaptures": 0}
    assert logged and logged[0].startswith("[MPPI-GRAPH] CUDA-graph MPPI capture failed")


def test_warmup_failure_fails_closed():
    backend = fx.ReplayBackend(fail_warmup=RuntimeError("CUDA error: out of memory"))
    _, inner, graph, _ = _pair(backend=backend)
    with pytest.raises(mc.GraphCaptureError, match="warmup failed .*out of memory"):
        fx.call(graph, fx.make_inputs("goal", N, 0))
    assert inner.control_steps == [0, 0] and graph.captured_graphs == 0


@pytest.mark.parametrize("break_inputs", [
    lambda i: dict(i, mode="bogus"),
    lambda i: dict(i, mode="dock"),                          # dock without dock_heading
    lambda i: dict(i, goal_radius=-1.0),
    lambda i: dict(i, state=i["state"][:, :5].contiguous()),
])
def test_invalid_inputs_raise_the_eager_value_error(break_inputs):
    eager, inner, graph, _ = _pair()
    inputs = break_inputs(fx.make_inputs("goal", N, 0))
    with pytest.raises(ValueError) as eager_error:
        fx.call(eager, inputs)
    with pytest.raises(ValueError) as graph_error:
        fx.call(graph, inputs)
    assert str(graph_error.value) == str(eager_error.value)
    assert not isinstance(graph_error.value, mc.GraphCaptureError) and inner.control_steps == [0, 0]


def test_graph_mode_refusals():
    dynamics, params = fx.frozen("blueboat_v2", K=16)
    legacy = mc.MPPIController(num_envs=N, params=dataclasses.replace(params, noise_backend=mc.LEGACY_NOISE_BACKEND),
                               dynamics=dynamics)
    with pytest.raises(mc.GraphCaptureError, match="keyed noise backend"):
        mc.CudaGraphMPPI(legacy, backend=fx.ReplayBackend())
    plain = mc.MPPIController(num_envs=N, params=params, dynamics=dynamics)
    with pytest.raises(mc.GraphCaptureError, match="needs a CUDA device"):
        mc.CudaGraphMPPI(plain)                              # CPU controller, real CUDA backend
    with pytest.raises(mc.GraphCaptureError, match="needs a CUDA device"):
        mc.CudaGraphBackend("cpu")
    with pytest.raises(mc.GraphCaptureError, match="has no noise_keys"):
        mc.CudaGraphMPPI(object(), backend=fx.ReplayBackend())
    graph = mc.CudaGraphMPPI(plain, backend=fx.ReplayBackend())
    inputs = fx.make_inputs("goal", N, 0)
    with pytest.raises(mc.GraphCaptureError, match="goal_radius must be a Python number"):
        fx.call(graph, dict(inputs, goal_radius=torch.tensor(2.0)))
    with pytest.raises(mc.GraphCaptureError, match="mode must be a Python str"):
        fx.call(graph, dict(inputs, mode=None))
    with pytest.raises(mc.GraphCaptureError, match="targets must be a tensor or None"):
        fx.call(graph, dict(inputs, targets=[[0.0, 0.0]]))
    with pytest.raises(TypeError):
        graph.rollout(inputs["state"], 64, targets=inputs["targets"], goal_radius=2.0)   # no K/H overrides


def test_reset_is_the_controller_reset_on_the_static_warm_start():
    eager, inner, graph, _ = _pair()
    _step(eager, graph, fx.make_inputs("goal", N, 0))
    static = graph._live.u_static
    graph.reset(torch.tensor([True, False]), [5, 0])
    eager.reset(torch.tensor([True, False]), [5, 0])
    assert inner.U is static and bool((static[0] == 0).all()) and not bool((static[1] == 0).all())
    assert inner.episode_ids == eager.episode_ids == [5, 0] and inner.control_steps == eager.control_steps == [0, 1]
    _step(eager, graph, fx.make_inputs("goal", N, 1))


def test_describe_and_stamps():
    eager, _, graph, _ = _pair()
    assert mc.eager_implementation_stamp() == {"mode": "eager", "captured_graphs": 0, "recaptures": 0}
    assert mc.MPPI_IMPLEMENTATIONS == ("eager", "cuda_graph_v1") and mc.MPPI_IMPL_CHOICES == ("auto", "eager", "graph")
    _step(eager, graph, fx.make_inputs("mission", N, 0))
    described = graph.describe()
    assert described["implementation"] == "cuda_graph_v1" and described["backend"] == "cpu_replay_stand_in"
    assert described["static_inputs"]["state"] == [N, 6] and described["static_inputs"]["gate_normals"] == [N, 1, 2]
    assert described["scalar_inputs"]["mode"] == "mission"
    assert graph.noise_metadata()["backend"] == mc.NOISE_BACKEND
    assert graph.params is eager.params and graph.num_envs == N and graph.dtype == torch.float32
