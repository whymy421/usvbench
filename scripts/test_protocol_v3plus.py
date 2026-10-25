"""D8 CPU regression tests; load only pure modules, never simulator packages."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protocol = load("eval_protocol")
paths = load("episode_path_observer")
fixtures = load("test_cert_v3")


def cfg():
    return NS(seed=0, eval_level=0, curriculum_frozen=False, episode_length_s=240.,
              drag_scale=1., sea_state=NS(hs_range=(0., 0.)),
              scene=NS(num_envs=2), sim=NS(device="cpu", dt=1/120), decimation=2)


def args(**extra):
    values = dict(eval_seed=42, level=1, num_envs=2, episodes=4, extra_sets=[],
                  device="cpu", allow_env=[], seed_namespace=None, seed_block=None,
                  speed_reference_m_s=None)
    values.update(extra)
    return NS(**values)


@pytest.mark.parametrize("item", ["seed=43", "eval_level=0", "curriculum_frozen=false",
                                  "scene.num_envs=1", "sim.device=cuda:0"])
def test_conflicting_protocol_overrides_rejected_before_launch(item):
    with pytest.raises(ValueError, match="conflict"):
        protocol.preflight_protocol(args(extra_sets=[item]), [])


def test_typed_nested_overrides_and_single_final_validation():
    config = cfg()
    snapshots = []
    config.validate_after_overrides = lambda: snapshots.append((config.eval_level, config.drag_scale))
    result = protocol.resolve_protocol(config, args(extra_sets=["sea_state.hs_range=0.2,0.4", "drag_scale=2.5"]))
    assert config.sea_state.hs_range == (.2, .4)
    assert config.curriculum_frozen and config.seed == 42
    assert result["eval_level"] == 1 and snapshots == [(1, 2.5)]


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_override_rejected(value):
    with pytest.raises(ValueError):
        protocol.resolve_protocol(cfg(), args(extra_sets=["drag_scale=" + value]))


def test_ambient_science_rejected_allowed_and_stamped():
    with pytest.raises(ValueError, match="GOAL_RADIUS_OVERRIDE"):
        protocol.ambient_environment([], {"GOAL_RADIUS_OVERRIDE": "3"})
    assert protocol.ambient_environment(["GOAL_RADIUS_OVERRIDE"],
                                        {"GOAL_RADIUS_OVERRIDE": "3", "PATH": "safe"}) == {"GOAL_RADIUS_OVERRIDE": "3"}
    with pytest.raises(ValueError, match="unknown"):
        protocol.ambient_environment(["OBS_UNDECLARED"], {"OBS_UNDECLARED": "1"})


def test_reserved_seed_cannot_be_used_as_diagnostic_or_selection():
    with pytest.raises(ValueError, match="namespace"):
        protocol.preflight_protocol(args(eval_seed=41041), [])
    with pytest.raises(ValueError):
        protocol.preflight_protocol(args(eval_seed=41041, seed_namespace="selection/1001",
                                         run_purpose="selection", seed_block="final-01"), [])


def test_pre_autoreset_full_and_success_prefix_paths():
    base = NS(num_envs=2, robot=NS(data=NS(root_pos_w=[[0., 0., 0.], [0., 0., 0.]])),
              _success=[False, False])
    expected = ([False, False], [True, True])
    base._get_dones = lambda: expected
    observer = paths.EpisodePathObserver(base)
    observer.after_reset()
    base.robot.data.root_pos_w = [[3., 4., 0.], [0., 2., 0.]]
    observer.observe(([False, False], [False, False]))
    base._success[0] = True
    base.robot.data.root_pos_w = [[6., 8., 0.], [0., 4., 0.]]
    observer.observe(([False, False], [False, False]))
    base.robot.data.root_pos_w = [[9., 12., 0.], [0., 6., 0.]]
    assert base._get_dones() is expected
    # Autoreset teleports; observer must never count this travel.
    base.robot.data.root_pos_w = [[100., 100., 0.], [200., 200., 0.]]
    base._success = [False, False]
    observer.after_step()
    assert observer.take(0, True) == {"path_length_full_episode_m": 15., "path_length_success_prefix_m": 10.}
    assert observer.take(1, False) == {"path_length_full_episode_m": 6., "path_length_success_prefix_m": None}
    base.robot.data.root_pos_w[0][0] += 1
    observer.observe(([True, False], [False, False]))
    observer.after_step()
    assert observer.take(0, False)["path_length_full_episode_m"] == 1.


def test_latency_deadline_and_resolved_horizon_are_certified():
    config = cfg()
    config.vehicle = "blueboat"
    config.curriculum_frozen = True
    base = NS(num_envs=2, device="cpu", termination_reason=[3, 3], control_step_s=1/60,
              max_episode_length=14400)
    ticks = iter([0., .020, .030, .031])
    run = fixtures.make_run(2, 2, base=base, env_cfg=config, overrides={}, clock=lambda: next(ticks),
                            cli_args={"speed_reference_m_s": 2.9, "speed_reference_basis": "nominal"})
    fixtures.step(run)
    records = [run.finish_episode(i, fixtures.sample_record(i)) for i in range(2)]
    doc = run.build({"records": records})
    assert doc["header"]["resolved_deadline_s"] == 240.
    assert doc["header"]["speed_reference"]["value_m_s"] == 2.9
    assert doc["header"]["controller_latency"]["deadline_misses_60hz"] == 1
    assert [r["controller_deadline_misses_60hz"] for r in records] == [1, 1]
    assert records[0]["episode_elapsed_s"] == 1/60


def test_runtime_curriculum_change_is_rejected():
    run = fixtures.make_run(2, 2)
    run._env_cfg.eval_level = 2
    with pytest.raises(RuntimeError, match="level"):
        fixtures.step(run)


def test_mppi_batch_and_episode_scheduling_do_not_change_noise_or_action():
    torch = pytest.importorskip("torch")
    mppi = load("mppi_controller")
    p = mppi.MPPIParams(K=8, H=2, seed=11)
    one = mppi.MPPIController(1, params=copy.deepcopy(p), env_ids=[7])
    many = mppi.MPPIController(2, params=copy.deepcopy(p), env_ids=[3, 7])
    for _ in range(2):
        assert torch.equal(one.sample_noise(8, 2)[0], many.sample_noise(8, 2)[1])
        a, _ = one.rollout(torch.zeros(1, 6), targets=torch.tensor([[[20., 0.]]]), goal_radius=2.)
        b, _ = many.rollout(torch.zeros(2, 6), targets=torch.tensor([[[20., 0.]], [[20., 0.]]]), goal_radius=2.)
        assert torch.equal(a[0], b[1])
    one.reset(torch.tensor([True]), episode_ids=[9])
    many.reset(torch.tensor([True, True]), episode_ids=[12, 9])
    assert torch.equal(one.sample_noise(8, 2)[0], many.sample_noise(8, 2)[1])
    prior = one.sample_noise(8, 2).clone()
    one.reset(torch.tensor([True]), episode_ids=[2])
    assert not torch.equal(prior, one.sample_noise(8, 2))
    one.reset(torch.tensor([True]), episode_ids=[9])
    assert torch.equal(prior, one.sample_noise(8, 2))
    assert one.noise_metadata()["backend"] == mppi.NOISE_BACKEND
    assert one.noise_metadata()["self_check"].startswith("passed:")


def test_no_isaac_imported():
    assert not any(key == "isaaclab" or key.startswith("isaaclab.") for key in sys.modules)
