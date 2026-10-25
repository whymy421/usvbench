"""CPU fixtures execute the real collectors without importing Isaac Lab."""
import ast
from contextlib import nullcontext
import copy
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


grid_module = load("fixed_grid")
intervals = load("paired_intervals")
FixedEpisodeGrid = grid_module.FixedEpisodeGrid


@pytest.mark.parametrize("episodes,envs", [(0, 2), (2, 0), (3, 2), (True, 1), (2, False), (2.0, 2)])
def test_invalid_or_rounded_budget_rejected(episodes, envs):
    with pytest.raises(ValueError):
        FixedEpisodeGrid(envs, episodes)


def test_completion_order_cannot_reweight_grid():
    grid = FixedEpisodeGrid(2, 4)
    accepted = [(i, ep) for i in [0, 0, 0, 1, 0, 1]
                if (ep := grid.accept(i)) is not None]
    assert set(accepted) == grid.expected == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert grid.counts == [2, 2] and grid.complete
    assert grid.accept(0) is None
    grid.require_complete()


def test_incomplete_population_is_an_error():
    grid = FixedEpisodeGrid(2, 4)
    grid.accept(0)
    with pytest.raises(RuntimeError, match="missing keys"):
        grid.require_complete()


class FakeEnvironment:
    def __init__(self, periods=(1, 3), drift=False):
        self.num_envs = len(periods)
        self.periods = periods
        self.max_episode_length = max(periods)
        self.tick = 0
        self.finished = [0] * self.num_envs
        self.device = "cpu"
        self.cfg = SimpleNamespace(eval_level=2, curriculum_frozen=True, goal_radius=1.0)
        self.curriculum_level = torch.full((self.num_envs,), 2)
        self.drift = drift
        self.phase = torch.zeros(self.num_envs, dtype=torch.long)
        self._reached_goal = torch.zeros(self.num_envs, dtype=torch.bool)
        self.episode_success = torch.zeros(self.num_envs, dtype=torch.bool)
        self.time_to_success = torch.full((self.num_envs,), math.nan)
        self.episode_path_length = torch.ones(self.num_envs)
        self.episode_min_clearance = torch.ones(self.num_envs)
        self.route_geodesic_length = torch.ones(self.num_envs)
        self.d0_per_env = torch.full((self.num_envs,), 10.0)
        self.episode_scenario_hashes = [{} for _ in periods]

    def _body_motion_observation(self):
        return torch.ones((self.num_envs, 1)), None, None

    def _horizontal_distance(self):
        return self.d0_per_env

    def step(self, actions):
        self.tick += 1
        done = torch.tensor([self.tick % period == 0 for period in self.periods])
        for i in torch.nonzero(done).flatten().tolist():
            self.episode_success[i] = i == 0
            self.time_to_success[i] = 1.0 if i == 0 else math.nan
            self.episode_scenario_hashes[i] = {"spawn_pose": f"{i}:{self.finished[i]}"}
            self.finished[i] += 1
            self.d0_per_env[i] += 1
        if self.drift:
            self.curriculum_level[0] = 3
        # Failure slot only truncates. Its actual terminated mask must survive.
        term = done & self.episode_success
        trunc = done & ~self.episode_success
        return torch.zeros((self.num_envs, 4)), None, term, trunc, {}


class FakeCertificate:
    """CertificateRun's collector API over the REAL FixedEpisodeGrid."""

    def __init__(self, num_envs, episodes):
        self.discarded = []
        self.steps = 0
        self.grid = FixedEpisodeGrid(num_envs, episodes)
        self.quota = self.grid.quota
        self.header = {"episodes_requested": episodes}

    def time_controller(self):
        return nullcontext()

    def time_environment(self):
        return nullcontext()

    def after_reset(self):
        pass

    def after_step(self):
        self.steps += 1

    def discard_episode(self, env):
        self.discarded.append(env)

    def next_episode(self, env):
        episode = self.grid.counts[env]
        if episode >= self.quota:
            self.discard_episode(env)
            return None
        return episode

    def finish_episode(self, env, record):
        assert record["terminated"] or record["truncated"]
        assert record["ep"] == self.grid.accept(env)
        return record


def collector_namespace(base):
    def act(obs, **kwargs):
        actions = torch.zeros((base.num_envs, 2))
        return actions, {"mean_actions": actions}
    runner = SimpleNamespace(agent=SimpleNamespace(act=act))
    cert = FakeCertificate(base.num_envs, 2 * base.num_envs)
    return dict(
        torch=torch, math=math, base=base, env=base, wrapped=base,
        assert_frozen_level=grid_module.assert_frozen_level,
        cert=cert, runner=runner, nav_runner=runner, dock_runner=runner,
        device=torch.device("cpu"), records=[], step=0,
        max_steps=(cert.quota + 3) * base.max_episode_length,
        args_cli=SimpleNamespace(episodes=2 * base.num_envs, num_envs=base.num_envs,
                                 level=2, switch_phase=2, nav_id="nav", dock_id="dock"),
        env_cfg=base.cfg, goal_radius=1.0, obs=torch.zeros((base.num_envs, 4)),
        observation=torch.zeros((base.num_envs, 4)),
        superset_observation=torch.zeros((base.num_envs, 4)),
        initial_distance=base.d0_per_env.clone(),
        episode_steps=torch.zeros(base.num_envs, dtype=torch.long),
        switch_steps=torch.full((base.num_envs,), -1, dtype=torch.long),
        _native_observation=lambda obs, _: obs, _champion_observation=lambda obs: obs,
        episode_scenario_hashes_for=lambda env, i: env.episode_scenario_hashes[i],
        completed=0, successes=0, failures_timeout=0, failures_other=0,
        active_steps=0, reverse_action_steps=0, reverse_velocity_steps=0,
        spl_values=[], success_times=[], path_lengths=[], min_clearances=[],
    )


def execute_collector(filename, base):
    tree = ast.parse((SCRIPTS / filename).read_text(encoding="utf-8"))
    scope = collector_namespace(base)
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name in ("_values_at", "_completed_path_lengths", "_initial_distances")]
    for node in helpers:
        node.decorator_list = []
    if filename == "convergence_curve.py":
        body = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "collect")
        nodes = helpers + [body]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), scope)
        # collect(target, obs) reads the rung's module-level ``cert``.
        scope["records"] = scope["collect"](4, scope["obs"])
    else:
        loop = next(n for n in ast.walk(tree) if isinstance(n, ast.While)
                    and "grid.complete" in ast.unparse(n.test))
        exec(compile(ast.Module(body=helpers + [loop], type_ignores=[]), filename, "exec"), scope)
    scope["cert"].grid.require_complete()
    return scope


@pytest.mark.parametrize("filename", ["eval_composite.py", "eval_hazard_nav.py",
                                     "eval_cross_champion.py", "convergence_curve.py"])
@pytest.mark.parametrize("periods", [(1, 3), (3, 1)])
def test_real_collector_loops_keep_slow_failures(filename, periods):
    scope = execute_collector(filename, FakeEnvironment(periods))
    records = scope["records"]
    assert {(r["env"], r["ep"]) for r in records} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert len(records) == 4
    assert sum(r["success"] for r in records) == 2
    assert all(r["truncated"] and not r["terminated"] for r in records if r["env"] == 1)
    assert scope["cert"].steps == 6
    assert len(scope["cert"].discarded) == 4
    assert {r["scenario_hashes"]["spawn_pose"] for r in records} == {"0:0", "0:1", "1:0", "1:1"}


def test_cross_champion_rejects_runtime_curriculum_drift():
    with pytest.raises(RuntimeError, match="runtime curriculum"):
        execute_collector("eval_cross_champion.py", FakeEnvironment(drift=True))


def test_paired_interval_uses_joint_table_and_has_nonzero_agreement_width():
    equal = intervals.paired_rate_interval([(True, True)] * 64)
    assert equal["difference"] == 0
    assert equal["interval"][0] < 0 < equal["interval"][1]
    forward = intervals.paired_rate_interval([(True, False)] * 20 + [(False, True)] * 5)
    reverse = intervals.paired_rate_interval([(False, True)] * 20 + [(True, False)] * 5)
    assert forward["difference"] == -reverse["difference"] == 0.6
    assert forward["interval"] == pytest.approx([-reverse["interval"][1], -reverse["interval"][0]])


def test_cpi_accepts_only_complete_certified_all_primitive_pairs():
    fixture = load("test_cert_v3")
    cpi = load("cpi_verdict")
    rung = fixture.document(2, 2)
    rung.update(episodes=4, successes=0, sr=0.0, step=1,
                opened_fresh=True, opened_episode_index=[0, 0],
                scenario_protocol={"version": "fixture-v1"})
    rung["header"]["pairing"] = {"unchanged_primitives": ["spawn_pose", "wave"],
                                  "changed_primitives": []}
    for r in rung["records"]:
        r["scenario_hashes"] = {"spawn_pose": f"{r['env']}:{r['ep']}", "wave": "same"}
    curve = {"paired": True, "pairing": "scenario-protocol", "rungs": [rung, copy.deepcopy(rung)]}
    assert len(cpi.checked_rungs(curve)) == 2
    for mutation in (lambda r: r["records"].pop(),
                     lambda r: r["records"].append(copy.deepcopy(r["records"][0])),
                     lambda r: r["records"][0]["scenario_hashes"].update(wave="different"),
                     lambda r: r.update(opened_fresh=False)):
        broken = copy.deepcopy(curve)
        mutation(broken["rungs"][1])
        with pytest.raises(ValueError):
            cpi.checked_rungs(broken)
