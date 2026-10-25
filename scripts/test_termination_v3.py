"""CPU-only tests for v3 stop-cause telemetry; no simulator or torch imports."""

import ast
import importlib.util
import itertools
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
HELPER = REPO / "tasks" / "_shared" / "termination_reason.py"
SPEC = importlib.util.spec_from_file_location("termination_reason_v3_cpu", HELPER)
REASONS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REASONS)

FAMILIES = {
    "blueboat_calm_nav": "blueboat_calm_nav_env.py",
    "boat_calm_nav": "my_first_task_env.py",
    "rov_calm_nav": "my_first_task_env.py",
    "path_following": "path_following_env.py",
    "path_hazard": "path_hazard_env.py",
    "hazard_nav": "hazard_nav_env.py",
    "harbor_mission": "harbor_mission_env.py",
    "station_keeping": "station_keeping_env.py",
    "station_keeping_boat": "station_keeping_boat_env.py",
    "docking": "docking_env.py",
}


class Vector:
    """Small masked vector implementing only the helper's public protocol."""

    def __init__(self, values):
        self.values = list(values)

    def long(self):
        return Vector(map(int, self.values))

    def clone(self):
        return Vector(self.values)

    def __and__(self, other):
        return Vector(a and b for a, b in zip(self.values, other.values))

    def __invert__(self):
        return Vector(not value for value in self.values)

    def __mul__(self, scalar):
        return Vector(value * scalar for value in self.values)

    def __setitem__(self, mask, value):
        for index, selected in enumerate(mask.values):
            if selected:
                self.values[index] = value

    def __lt__(self, value):
        return Vector(item < value for item in self.values)


class TerminationReasonTests(unittest.TestCase):
    def test_enum_contract(self):
        self.assertEqual(REASONS.TERMINATION_REASON_NAMES, {
            0: "running", 1: "success", 2: "collision", 3: "timeout",
            4: "out_of_bounds", 5: "failure",
        })

    def test_all_mask_combinations_and_no_input_mutation(self):
        cases = list(itertools.product((False, True), repeat=5))
        vectors = [Vector(row[col] for row in cases) for col in range(5)]
        before = [vector.values[:] for vector in vectors]
        result = REASONS.termination_reason_codes(
            vectors[0], vectors[1], success=vectors[2], collision=vectors[3],
            out_of_bounds=vectors[4],
        )
        expected = []
        for terminated, truncated, success, collision, out_of_bounds in cases:
            if not terminated:
                expected.append(3 if truncated else 0)
            elif out_of_bounds:
                expected.append(4)
            elif collision:
                expected.append(2)
            elif success:
                expected.append(1)
            else:
                expected.append(5)
        self.assertEqual(result.values, expected)
        self.assertEqual([vector.values for vector in vectors], before)

    def test_fixed_horizon_success_and_contact_remain_timeouts(self):
        reasons = REASONS.termination_reason_codes(
            Vector([False, False]), Vector([True, True]),
            success=Vector([True, False]), collision=Vector([False, True]),
        )
        self.assertEqual(reasons.values, [3, 3])

    def test_publish_clone_and_next_step_running(self):
        env = SimpleNamespace(extras={})
        REASONS.publish_termination_reason(
            env, Vector([True, False]), Vector([False, True]),
            success=Vector([True, False]),
        )
        saved = env.extras["termination_reason"]
        self.assertEqual(saved.values, [1, 3])
        env.termination_reason.values[0] = 5
        self.assertEqual(saved.values, [1, 3])
        REASONS.publish_termination_reason(
            env, Vector([False, False]), Vector([False, False])
        )
        self.assertEqual(env.termination_reason.values, [0, 0])
        self.assertEqual(saved.values, [1, 3])

    def test_all_families_publish_before_return_and_do_not_reset_reason(self):
        for family, filename in FAMILIES.items():
            with self.subTest(family=family):
                tree = ast.parse((REPO / "tasks" / family / filename).read_text(
                    encoding="utf-8"
                ))
                methods = {
                    node.name: node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                }
                body = methods["_get_dones"].body
                self.assertIsInstance(body[-1], ast.Return)
                call = body[-2].value
                self.assertIsInstance(call, ast.Call)
                self.assertEqual(call.func.id, "publish_termination_reason")
                # The same exact expressions passed into telemetry are returned.
                self.assertEqual(
                    [ast.dump(arg) for arg in call.args[1:3]],
                    [ast.dump(arg) for arg in body[-1].value.elts],
                )
                reset = methods["_reset_idx"]
                self.assertFalse(any(
                    isinstance(node, ast.Attribute) and node.attr == "termination_reason"
                    for node in ast.walk(reset)
                ))

    def test_hazard_call_does_not_misclassify_contacted_arrival(self):
        path = REPO / "tasks" / "hazard_nav" / "hazard_nav_env.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        method = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef) and node.name == "_get_dones")
        publish = ast.Module(body=[method.body[-2]], type_ignores=[])
        for contact_terminates, expected in ((False, 5), (True, 2)):
            env = SimpleNamespace(
                extras={}, _success=Vector([False]),
                cfg=SimpleNamespace(contact_terminates=contact_terminates),
            )
            exec(compile(publish, str(path), "exec"), {
                "self": env, "terminated": Vector([True]),
                "time_out": Vector([False]), "contact_now": Vector([True]),
                "publish_termination_reason": REASONS.publish_termination_reason,
            })
            self.assertEqual(env.termination_reason.values, [expected])


if __name__ == "__main__":
    unittest.main()
