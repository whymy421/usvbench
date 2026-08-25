"""No-Isaac proof that EVERY certificate writer stamps the scenario protocol.

WHY THIS FILE EXISTS
--------------------
``tasks/_shared/scenario_rng.py`` makes an episode scenario a pure function of
``(protocol_version, eval_seed, env_index, episode_index, group)``, and
``tasks/_shared/scenario_draws.py:stamp_scenario`` digests the RESOLVED
parameters so two runs can be checked for having sat the same exam paper.  That
machinery is worth nothing to the comparisons that actually matter -- the ones
BETWEEN controllers -- unless every evaluator writes the digests down.  Before
this file, only ``scripts/eval_v6_frozen.py`` did, so an MPPI or classical
certificate could never be scenario-paired against a policy certificate; it
could only be ASSUMED comparable, which is the assumption the whole scenario-RNG
fix exists to stop making.

WHAT IT CHECKS
--------------
1. Source hygiene for every file the stamping touched: AST-parses, pure ASCII,
   no control characters other than LF/CR.
2. The shared helpers in ``tasks/_shared/scenario_draws.py`` behave, against a
   FABRICATED env object: header when the env carries ``_scenario``, the
   off-protocol literal when it does not, a defensive copy of the latched
   digests, and ``None`` (meaning "write no field") for the three not-a-stamped
   episode cases.
3. No drift between the shared literal and the copy ``scripts/eval_v6_frozen.py``
   keeps at :91.
4. THE REAL ONE: each evaluator's own record-assembly loop is lifted out of its
   source by AST -- statement for statement, not a copy -- and executed against
   the fabricated env.  The record it produces must carry ``scenario_hashes``
   when the env latched digests, and must NOT carry the key on a family that is
   not on the protocol.  This is the same "the test and the shipped code run the
   SAME statements" trick ``scripts/test_classical_baseline_math.py:110-125``
   uses for the PID math: the evaluators import ``isaaclab`` (or launch the app)
   at module scope and cannot be imported on a machine without Isaac Sim.
5. Each certificate JSON header carries ``scenario_protocol`` beside its records.

Run:  python scripts/test_certificate_scenario_stamp.py
  or: python -m pytest scripts/test_certificate_scenario_stamp.py -v
"""

from __future__ import annotations

import ast
import math
import os
import sys

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_SCRIPTS)
for _path in (_REPO, _SCRIPTS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from tasks._shared.scenario_draws import (  # noqa: E402
    SCENARIO_PROTOCOL_OFF,
    episode_scenario_hashes_for,
    scenario_protocol_header,
    scenario_protocol_notice,
    scenario_protocol_stamp,
)

# Every script under scripts/ that writes a JSON carrying a "records" list.
# The value is the sibling key that identifies the certificate header dict --
# "records" for the flat certificates, "rungs" for the checkpoint-ladder sweep,
# whose per-episode records are nested one level down inside each rung.
CERT_WRITERS = {
    "eval_v6_frozen.py": "records",
    "eval_mppi.py": "records",
    "classical_baseline.py": "records",
    "classical_planner_pid.py": "records",
    "eval_imbalance.py": "records",
    "eval_obs_bridge.py": "records",
    "eval_composite.py": "records",
    "diagnose_failures.py": "records",
    "convergence_curve.py": "rungs",
}

# eval_v6_frozen.py is the writer the helpers were factored out of; it keeps its
# own inline guard (see test_no_drift_against_eval_v6_frozen), every other
# writer calls the shared helpers.
HELPER_CALLERS = tuple(name for name in CERT_WRITERS if name != "eval_v6_frozen.py")

FROZEN_EVAL = os.path.join(_SCRIPTS, "eval_v6_frozen.py")
SCENARIO_DRAWS = os.path.join(_REPO, "tasks", "_shared", "scenario_draws.py")
SELF = os.path.abspath(__file__)

TOUCHED_SOURCES = (
    [SCENARIO_DRAWS, SELF]
    + [os.path.join(_SCRIPTS, name) for name in sorted(CERT_WRITERS)]
)

# diagnose_failures.py prints its failure taxonomy in Chinese on purpose (its
# module docstring documents the cp1252 crash that forced the stdout
# reconfigure), so it is 312 non-ASCII bytes at HEAD and exempt from the ASCII
# half of the hygiene check ONLY. The control-character half still applies.
NON_ASCII_BY_DESIGN = ("diagnose_failures.py",)


# ---------------------------------------------------------------- layer 1


def test_source_hygiene():
    """No stray control character or non-ASCII byte in anything I touched."""
    for target in TOUCHED_SOURCES:
        data = open(target, "rb").read()
        ascii_exempt = os.path.basename(target) in NON_ASCII_BY_DESIGN
        bad = [
            (index, byte)
            for index, byte in enumerate(data)
            if (byte < 32 and byte not in (10, 13))
            or (byte > 126 and not ascii_exempt)
        ]
        assert not bad, f"control/non-ascii bytes in {target}: {bad[:5]}"
        ast.parse(data.decode("utf-8"), filename=target)


# ---------------------------------------------------------------- layer 2
# A fabricated env carrying exactly the attributes an evaluator reads out of
# the real Isaac env when an episode ends. Plain lists, because every read in
# every record loop is `float(...)`, `int(...)`, `bool(...)` or a subscript.

LATCHED_HASHES = (
    {"spawn_pose": "1111", "current": "2222", "scenario": "3333"},
    {"spawn_pose": "4444", "current": "5555", "scenario": "6666"},
)


class FakeScenario:
    """Stand-in for the ScenarioRNG the migrated envs hang off ``_scenario``."""


class FakeEnv:
    """One finished-episode snapshot for two envs.

    ``on_protocol=False`` is a family that was never migrated: no ``_scenario``
    and no ``episode_scenario_hashes`` at all, exactly like the certificates
    written before the scenario protocol existed.
    """

    def __init__(self, on_protocol=True, latched=True):
        self.num_envs = 2
        self.control_step_s = 1.0 / 60.0
        self.episode_success = [True, False]
        self.time_to_success = [12.5, float("nan")]
        self.episode_path_length = [30.0, 44.0]
        self.episode_min_clearance = [0.8, -0.1]
        self.episode_xte_rms = [0.4, 0.9]
        self.episode_contact_steps = [0.0, 3.0]
        self.episode_contact_longest_steps = [0.0, 2.0]
        self.episode_contact_depth_sum = [0.0, 0.15]
        self.episode_max_phase = [3, 1]
        self.episode_gates_passed = [4, 2]
        self.route_geodesic_length = [28.0, 41.0]
        if on_protocol:
            self._scenario = FakeScenario()
            self.episode_scenario_hashes = (
                [dict(entry) for entry in LATCHED_HASHES]
                if latched
                else [{} for _ in range(self.num_envs)]
            )


def test_protocol_stamp_is_the_header_only_when_the_env_carries_it():
    assert scenario_protocol_stamp(FakeEnv()) == scenario_protocol_header()
    # Never migrated: no attribute at all.
    assert scenario_protocol_stamp(FakeEnv(on_protocol=False)) == \
        SCENARIO_PROTOCOL_OFF
    # Migrated but built unseeded: make_scenario_rng returned None, so every
    # draw fell back to the historical global-RNG line. Off protocol too.
    unseeded = FakeEnv()
    unseeded._scenario = None
    assert scenario_protocol_stamp(unseeded) == SCENARIO_PROTOCOL_OFF
    # The marker must stay a plain string: an auditor that reads it as a dict
    # would treat an off-protocol run as compliant.
    assert isinstance(SCENARIO_PROTOCOL_OFF, str)
    assert isinstance(scenario_protocol_stamp(FakeEnv()), dict)


def test_episode_hashes_are_a_defensive_copy():
    env = FakeEnv()
    got = episode_scenario_hashes_for(env, 0)
    assert got == dict(LATCHED_HASHES[0])
    # A later reset rewrites the latch; a record already appended to the
    # certificate must not follow it.
    env.episode_scenario_hashes[0]["spawn_pose"] = "OVERWRITTEN"
    assert got["spawn_pose"] == "1111"


def test_episode_hashes_return_none_for_the_three_unstamped_cases():
    # 1. family not on the protocol -> no attribute
    assert episode_scenario_hashes_for(FakeEnv(on_protocol=False), 0) is None
    # 2. env has not finished its first episode -> latch still the empty dict
    assert episode_scenario_hashes_for(FakeEnv(latched=False), 0) is None
    # 3. an env index the container does not carry
    assert episode_scenario_hashes_for(FakeEnv(), 99) is None


def test_protocol_notice_text():
    assert scenario_protocol_notice("Isaac-USV-X", scenario_protocol_header()) is None
    notice = scenario_protocol_notice("Isaac-USV-X", SCENARIO_PROTOCOL_OFF)
    assert notice is not None
    assert "Isaac-USV-X" in notice
    assert SCENARIO_PROTOCOL_OFF in notice
    assert "NOT paired across evaluation seeds" in notice


def test_cross_controller_pairing_key_is_controller_free():
    """The digests two controllers read are the same object, by construction.

    The record field is copied straight off the env's per-episode latch, so at
    a fixed (eval seed, env, episode) an MPPI run and a policy run write the
    same digest. Nothing about the controller enters it -- which is precisely
    why ``stamp_scenario`` hashes VALUES and never the key
    (tasks/_shared/scenario_draws.py, "SCENARIO HASHES ARE OVER VALUES").
    """
    env = FakeEnv()
    policy_side = episode_scenario_hashes_for(env, 1)
    mppi_side = episode_scenario_hashes_for(env, 1)
    assert policy_side == mppi_side == dict(LATCHED_HASHES[1])
    assert policy_side is not mppi_side


# ---------------------------------------------------------------- layer 3


def test_no_drift_against_eval_v6_frozen():
    """eval_v6_frozen.py keeps its own copy of the marker; pin them equal."""
    tree = ast.parse(open(FROZEN_EVAL, encoding="utf-8").read(),
                     filename=FROZEN_EVAL)
    literals = [
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and any(
            isinstance(target, ast.Name) and target.id == "SCENARIO_PROTOCOL_OFF"
            for target in node.targets
        )
    ]
    assert literals == [SCENARIO_PROTOCOL_OFF], (
        f"eval_v6_frozen.py's SCENARIO_PROTOCOL_OFF is {literals!r} but the "
        f"shared literal is {SCENARIO_PROTOCOL_OFF!r}; a certificate written "
        "by one writer would then be unpairable with the other's"
    )


# ---------------------------------------------------------------- layer 4


def _module_tree(name):
    path = os.path.join(_SCRIPTS, name)
    return path, ast.parse(open(path, encoding="utf-8").read(), filename=path)


def _record_loop(name):
    """The innermost ``for`` whose body appends to ``records``, by AST."""
    path, tree = _module_tree(name)
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "append"
            and isinstance(inner.func.value, ast.Name)
            and inner.func.value.id == "records"
            for inner in ast.walk(node)
        )
    ]
    innermost = [
        node
        for node in candidates
        if not any(
            other is not node and other in ast.walk(node) for other in candidates
        )
    ]
    assert len(innermost) == 1, (
        f"{name}: expected exactly one record-assembly loop, found "
        f"{len(innermost)} (of {len(candidates)} candidates)"
    )
    return path, innermost[0]


def _namespace(env, env_index):
    """Everything an evaluator's record loop reads, other than the loop var.

    Deliberately a superset: a NameError here means an evaluator grew a new
    per-episode read and this fabrication has to grow with it, which is a
    failure worth seeing rather than papering over.
    """
    return {
        "math": math,
        "base": env,
        "records": [],
        "ep_counter": [0] * env.num_envs,
        "d0_prev": [29.5, 40.0],
        "switch_steps": [55, -1],                  # eval_composite
        "acc_steps": [180.0, 240.0],               # diagnose_failures
        "acc_min_goal": [1.2, 9.0],
        "acc_engaged": [30.0, 0.0],
        "acc_engaged_last": [120.0, -1.0],
        "acc_fwd": [150.0, 200.0],
        "acc_speed": [220.0, 90.0],
        "is_path_follow": False,                   # classical_baseline
        "zone_radius": 2.0,
        "spl_values": [],
        "success_times": [],
        "successes": 0,
        "failures_timeout": 0,
        "failures_other": 0,
        "episode_scenario_hashes_for": episode_scenario_hashes_for,
    }


def _loop_variables(name, env, env_index):
    """Bind the loop target(s) for the one iteration under test."""
    if name == "classical_baseline.py":
        # for env_id, success, timed_out, terminated_flag,
        #     shortest_path_source, path_length, time_s in zip(...)
        return {
            "env_id": env_index,
            "success": bool(env.episode_success[env_index]),
            "timed_out": False,
            "terminated_flag": True,
            "shortest_path_source": 28.0,
            "path_length": float(env.episode_path_length[env_index]),
            "time_s": float(env.time_to_success[env_index]),
        }
    return {"i": env_index}


def _run_record_loop(name, env, env_index=0):
    """Execute the evaluator's OWN record-assembly statements once."""
    path, loop = _record_loop(name)
    namespace = _namespace(env, env_index)
    namespace.update(_loop_variables(name, env, env_index))
    body = ast.Module(body=loop.body, type_ignores=[])
    exec(compile(body, path, "exec"), namespace)
    records = namespace["records"]
    assert len(records) == 1, f"{name}: loop produced {len(records)} records"
    return records[0]


def test_every_evaluator_emits_scenario_hashes():
    """The point of the whole exercise, one evaluator at a time."""
    for name in sorted(CERT_WRITERS):
        record = _run_record_loop(name, FakeEnv(), env_index=0)
        assert "scenario_hashes" in record, (
            f"{name}: the record-assembly loop dropped scenario_hashes, so its "
            "certificates cannot be scenario-paired against any other "
            "controller's"
        )
        assert record["scenario_hashes"] == dict(LATCHED_HASHES[0]), (
            f"{name}: stamped {record['scenario_hashes']!r}"
        )
        # The whole-scenario digest is what pairs an episode across
        # controllers; a per-primitive-only stamp would not.
        assert "scenario" in record["scenario_hashes"]


def test_every_evaluator_emits_the_same_hashes_for_the_same_episode():
    """Cross-controller pairing: same (env, episode) -> byte-identical field."""
    stamped = {
        name: _run_record_loop(name, FakeEnv(), env_index=1)["scenario_hashes"]
        for name in sorted(CERT_WRITERS)
    }
    distinct = {tuple(sorted(value.items())) for value in stamped.values()}
    assert len(distinct) == 1, (
        "evaluators disagree on the digests of the same episode: "
        f"{stamped!r}"
    )
    assert stamped["eval_mppi.py"] == dict(LATCHED_HASHES[1])


def test_off_protocol_family_gets_no_hash_field():
    """A family never migrated must stay silent, not stamp an empty dict."""
    for name in sorted(CERT_WRITERS):
        record = _run_record_loop(name, FakeEnv(on_protocol=False), env_index=0)
        assert "scenario_hashes" not in record, (
            f"{name}: stamped scenario_hashes on an env with no scenario "
            "protocol; a reader would take that as compliance"
        )


def test_first_episode_before_any_latch_gets_no_hash_field():
    for name in sorted(CERT_WRITERS):
        record = _run_record_loop(name, FakeEnv(latched=False), env_index=0)
        assert "scenario_hashes" not in record, (
            f"{name}: stamped an empty latch as if it were a scenario"
        )


# ---------------------------------------------------------------- layer 5


def _dict_key_sets(tree):
    """String-key sets of every dict literal in the module."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            out.append({
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            })
    return out


def test_every_certificate_header_carries_scenario_protocol():
    for name, marker in sorted(CERT_WRITERS.items()):
        _path, tree = _module_tree(name)
        headers = [keys for keys in _dict_key_sets(tree) if marker in keys]
        assert headers, f"{name}: no dict literal carrying {marker!r}"
        assert any("scenario_protocol" in keys for keys in headers), (
            f"{name}: the certificate header next to {marker!r} has no "
            "scenario_protocol field, so a reader cannot tell whether the run "
            "was on the protocol at all"
        )


def test_every_other_writer_calls_the_shared_helpers():
    """One implementation of the guard, not nine near-copies."""
    for name in sorted(HELPER_CALLERS):
        _path, tree = _module_tree(name)
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        for helper in ("scenario_protocol_stamp", "episode_scenario_hashes_for",
                       "scenario_protocol_notice"):
            assert helper in called, f"{name} never calls {helper}()"


def test_record_field_name_is_the_one_the_auditor_reads():
    """scripts/check_scenario_independence.py:302 reads exactly this key."""
    auditor = os.path.join(_SCRIPTS, "check_scenario_independence.py")
    source = open(auditor, encoding="utf-8").read()
    assert 'record.get("scenario_hashes")' in source
    assert 'payload.get("scenario_protocol")' in source


# ---------------------------------------------------------------- runner


def main() -> int:
    failures = 0
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - report, do not abort the run
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {name}")
    print(f"test_certificate_scenario_stamp: {len(tests)} tests, "
          f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
