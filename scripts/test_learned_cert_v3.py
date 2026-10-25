"""CPU-only execution of the learned writers' real rollout loops.

AST extraction avoids importing Isaac Lab or torch. Small vector/environment
fakes deliberately complete at different rates to expose pooled sampling.
"""

import ast
from contextlib import contextmanager, nullcontext
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


WRITERS = ("eval_v6_frozen.py", "eval_imbalance.py", "eval_obs_bridge.py")


class Vector:
    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, index):
        return self.values[index]

    def __setitem__(self, index, value):
        self.values[index] = value

    def __lt__(self, value):
        return Vector(item < value for item in self.values)

    def __ge__(self, value):
        return Vector(item >= value for item in self.values)

    def __or__(self, other):
        return Vector(a or b for a, b in zip(self.values, other.values))

    def any(self):
        return any(self.values)

    def all(self):
        return all(self.values)

    def clone(self):
        return Vector(self.values)

    def dim(self):
        return 1

    def flatten(self):
        return self

    def reshape(self, *_shape):
        return self

    def tolist(self):
        return list(self.values)


class VectorEnv:
    def __init__(self, periods):
        self.periods = periods
        self.steps = 0
        self.num_envs = len(periods)
        self.max_episode_length = 10
        self.episode_success = [True, False, True]
        self.time_to_success = [1.0, float("nan"), 4.0]
        self.episode_path_length = [1.0, 2.0, 4.0]
        self.d0_per_env = Vector([10.0, 20.0, 30.0])

    def reset(self):
        return 0, None

    def step(self, _actions):
        self.steps += 1
        done = [self.steps % period == 0 for period in self.periods]
        for index, completed in enumerate(done):
            if completed:
                # Reproduce the automatic reset that replaces live d0.
                self.d0_per_env[index] += 100.0
        return (
            self.steps, None,
            Vector([done[0], False, done[2]]),
            Vector([False, done[1], done[2]]), None,
        )


class CertificateProbe:
    def __init__(self, count):
        self.pending = [0] * count
        self.completed = []
        self.discarded = []
        self.controller_calls = 0
        self.environment_calls = 0
        self.after_steps = 0
        self.reset_calls = 0
        self.phase = None
        self.events = []

    def after_reset(self):
        assert self.after_steps == 0
        self.reset_calls += 1

    @contextmanager
    def time_controller(self):
        assert self.phase is None
        self.phase = "controller"
        self.controller_calls += 1
        self.events.append("controller")
        yield
        self.phase = None

    @contextmanager
    def time_environment(self):
        assert self.phase is None
        self.phase = "environment"
        self.environment_calls += 1
        self.events.append("environment")
        yield
        self.phase = None

    def after_step(self):
        assert self.phase is None
        assert self.controller_calls == self.environment_calls == self.after_steps + 1
        self.after_steps += 1
        self.events.append("after_step")
        self.pending = [count + 1 for count in self.pending]

    def finish_episode(self, index, record):
        assert self.pending[index] > 0
        self.completed.append((index, record["ep"], self.pending[index]))
        self.pending[index] = 0
        return dict(record)

    def discard_episode(self, index):
        self.discarded.append(index)
        self.pending[index] = 0


def run_rollout(writer, periods=(1, 2, 4), requested=6):
    path = Path(__file__).with_name(writer)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {"ep_counter", "quota", "max_steps", "step", "grid_complete"}
    statements = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            if (isinstance(call.func, ast.Attribute) and call.func.attr == "reset"
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == "wrapped"):
                statements.append(node)
                continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if (isinstance(call.func, ast.Attribute) and call.func.attr == "after_reset"
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == "cert"):
                statements.append(node)
                continue
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names
            for target in node.targets
        ):
            statements.append(node)
        elif isinstance(node, ast.While):
            statements.append(node)
    assert sum(isinstance(node, ast.While) for node in statements) == 1
    env = VectorEnv(periods)
    cert = CertificateProbe(env.num_envs)
    from fixed_grid import episode_quota
    cert.quota = episode_quota(requested, env.num_envs)
    bridge_calls = []

    def act(obs, **_kwargs):
        assert cert.phase == "controller"
        return [obs, {"mean_actions": obs}]

    def bridge(obs):
        assert cert.phase == "controller", "bridge work must be timed"
        bridge_calls.append(obs)
        return obs

    def step(actions):
        assert cert.phase == "environment"
        return env.step(actions)

    namespace = {
        "args_cli": SimpleNamespace(episodes=requested),
        "base": env, "cert": cert, "math": math, "obs": 0, "records": [],
        "runner": SimpleNamespace(agent=SimpleNamespace(act=act)),
        "wrapped": SimpleNamespace(step=step, reset=env.reset), "_bridge_obs": bridge,
        "episode_scenario_hashes_for": lambda _base, _index: None,
        "episode_appearance_fields_for": lambda _base, _index: {},
        # D1: C4 condition latch helper called by eval_v6_frozen.py.
        "episode_certificate_fields": lambda _base, _index: {},
        "torch": SimpleNamespace(
            long=int, inference_mode=nullcontext,
            zeros=lambda count, **_kwargs: Vector([0] * count),
            nonzero=lambda values: Vector(
                index for index, value in enumerate(values.values) if value
            ),
        ),
    }
    module = ast.Module(body=statements, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace, env, cert, bridge_calls


class LearnedCertificateTests(unittest.TestCase):
    def test_asynchronous_completion_retains_equal_per_env_quota(self):
        for writer in WRITERS:
            with self.subTest(writer=writer):
                result, env, cert, bridge = run_rollout(writer)
                self.assertEqual(result["quota"], 2)
                self.assertEqual(
                    {(rec["env"], rec["ep"]) for rec in result["records"]},
                    {(env_id, ep) for env_id in range(3) for ep in range(2)},
                )
                self.assertEqual(len(result["records"]), 6)
                self.assertTrue(result["grid_complete"])
                self.assertEqual(cert.reset_calls, 1)
                self.assertEqual(env.steps, 8)
                self.assertEqual(len(cert.discarded), 8)
                self.assertEqual(cert.after_steps, env.steps)
                self.assertEqual(cert.events, ["controller", "environment", "after_step"] * 8)
                self.assertEqual(len(bridge), 8 if "bridge" in writer else 0)
                for env_id, _ep, steps in cert.completed:
                    self.assertEqual(steps, (1, 2, 4)[env_id])

    def test_real_done_masks_and_pre_reset_d0_are_preserved(self):
        for writer in WRITERS:
            with self.subTest(writer=writer):
                result, _env, _cert, _bridge = run_rollout(writer)
                for rec in result["records"]:
                    env_id = rec["env"]
                    self.assertEqual(rec["terminated"], env_id in (0, 2))
                    self.assertEqual(rec["truncated"], env_id in (1, 2))
                    self.assertEqual(rec["d0_m"], (10, 20, 30)[env_id] + 100 * rec["ep"])

    def test_step_budget_marks_incomplete_without_pooled_replacement(self):
        for writer in WRITERS:
            with self.subTest(writer=writer):
                result, env, _cert, _bridge = run_rollout(writer, periods=(1, 2, 100))
                self.assertFalse(result["grid_complete"])
                self.assertEqual(env.steps, result["max_steps"])
                self.assertEqual(len(result["records"]), 4)
                self.assertEqual({rec["env"] for rec in result["records"]}, {0, 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
