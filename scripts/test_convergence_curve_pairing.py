"""No-Isaac proof that the checkpoint-ladder sweep really pairs its rungs.

WHY THIS FILE EXISTS
--------------------
``scripts/convergence_curve.py`` compares checkpoints of one run at a time and
the whole point of the comparison is that every rung sits the SAME exam, so a
rung-to-rung difference is policy and not layout.  Two separate mechanisms have
to hold for that to be true, and each one used to be able to fail silently while
the curve JSON still said ``paired: true``:

1. THE SCENARIO STREAM.  ``restart_scenarios`` rebuilds the ScenarioRNG so the
   per-env episode counters go back to "never reset".  Rebuilding that object
   does NOT touch ``_suite_s_episode_counter``, the SECOND per-env counter the
   five Suite S ids use to rotate through their frozen layout library
   (``tasks/hazard_nav/hazard_nav_env.py:2074-2080``).  On those five ids rung
   k+1 therefore carried on from wherever rung k stopped and env i's first
   episode faced layout ``(i + k) % 10`` instead of ``(i + 0) % 10``.  Since
   ``scripts/cpi_verdict.py:51-56`` pairs rungs on each env's FIRST record, that
   drift lands exactly on the records the verdict is computed from.

2. THE RUNG OPENING.  The rung used to open with ``wrapped.reset()``.  skrl's
   Isaac Lab wrapper is understood to reset the underlying env only on its first
   call, which would leave every rung after the first opening on an episode
   carried over from the previous checkpoint -- drawn under the pre-rewind
   counters and steered, for its first steps, by the previous policy.  The sweep
   now opens each rung with a reset of the UNWRAPPED env, which resets
   unconditionally (``isaaclab/envs/direct_rl_env.py:292-331``), so both
   possible wrapper behaviours produce the same rung.

WHAT IT CHECKS
--------------
The functions under test are lifted out of ``convergence_curve.py`` BY AST and
executed here -- the same trick ``scripts/test_certificate_scenario_stamp.py``
uses -- because that script launches Isaac Sim at import time and cannot be
imported on a machine without it.  So these are the shipped statements running,
not a copy of them.

* The Suite S rotation is replayed with the env's OWN two statements and the
  real ``rotation_index``, over several rungs, and the layout each env draws
  must be identical rung for rung.  A fixture that could not see the drift
  would be worthless, so one test deliberately restarts the protocol object
  ALONE and asserts the rungs come apart.
* ``--unpaired`` must leave the rotation running: that flag exists to reproduce
  screen_v6_ladder, and a rewind that ignored it would quietly change what the
  flag means.
* The rung opening must not go back through the wrapper, and the curve JSON must
  keep carrying the fields that let a reader check all of this after the fact.

Run:  python scripts/test_convergence_curve_pairing.py
  or: python -m pytest scripts/test_convergence_curve_pairing.py -v
"""

from __future__ import annotations

import ast
import importlib.util
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_SCRIPTS)
for _path in (_REPO, _SCRIPTS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from tasks._shared.scenario_draws import (  # noqa: E402
    episode_scenario_hashes_for,
    make_scenario_rng,
)

SCRIPT = os.path.join(_SCRIPTS, "convergence_curve.py")
SOURCE = open(SCRIPT, encoding="utf-8").read()
SELF = os.path.abspath(__file__)


def _load_rotation_index():
    """The env's real layout picker, without importing the gym registrations.

    ``tasks/hazard_nav/__init__.py`` imports gymnasium at module scope, so the
    package path is closed on a machine with no Isaac stack; the module itself
    only needs numpy and torch.  Loading it by file path is what keeps this test
    honest -- the rotation rule asserted below is the one the env runs.
    """
    package_dir = os.path.join(_REPO, "tasks", "hazard_nav")
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)
    spec = importlib.util.spec_from_file_location(
        "usvbench_suite_s_layouts_direct",
        os.path.join(package_dir, "suite_s_layouts.py"),
    )
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules during class
    # creation, so the module has to be registered before it is executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.rotation_index, len(module.SUITE_S_PUBLIC_INDICES)


ROTATION_INDEX, LAYOUT_COUNT = _load_rotation_index()

# What each test measured, printed by the standalone runner beside its PASS.
# A collector rather than a return value: pytest 9 warns on a test that returns
# something, and these notes are worth keeping -- "3 rungs identical" in a log
# is the evidence, "PASS" on its own is only a claim.
NOTES: list[str] = []


def note(text):
    NOTES.append(text)


# ---------------------------------------------------------------- layer 1


def test_source_hygiene():
    """No stray control character or non-ASCII byte in either file."""
    for target in (SCRIPT, SELF):
        data = open(target, "rb").read()
        bad = [
            (index, byte)
            for index, byte in enumerate(data)
            if (byte < 32 and byte not in (10, 13)) or byte > 126
        ]
        assert not bad, f"control/non-ascii bytes in {target}: {bad[:5]}"
        ast.parse(data.decode("utf-8"), filename=target)


# ---------------------------------------------------------------- layer 2
# The sweep's own functions, lifted by AST and executed here.


def _script_tree():
    return ast.parse(SOURCE, filename=SCRIPT)


def _function_def(name):
    found = [
        node
        for node in _script_tree().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(found) == 1, f"{name}: found {len(found)} definitions"
    return found[0]


def load_sweep_namespace(unpaired=False, eval_seed=42):
    """Execute every top-level def and literal constant of the sweep script.

    Everything else at module scope -- the argparse block, the AppLauncher, the
    gym.make, the ladder loop -- is skipped, so nothing imports Isaac Sim.  The
    globals those functions read are supplied here instead.
    """
    body = []
    for node in _script_tree().body:
        if isinstance(node, ast.FunctionDef):
            body.append(node)
            continue
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                continue
            body.append(node)
    namespace = {
        "np": np,
        "torch": torch,
        "math": math,
        "make_scenario_rng": make_scenario_rng,
        "episode_scenario_hashes_for": episode_scenario_hashes_for,
        "args_cli": SimpleNamespace(unpaired=unpaired, eval_seed=eval_seed),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), SCRIPT, "exec"),
         namespace)
    assert "restart_scenarios" in namespace
    assert "restart_suite_s_rotation" in namespace
    return namespace


class FakeSuiteSEnv:
    """A Suite S env reduced to the two counters a rung restart has to rewind.

    ``finish_episode`` runs the same two statements the real env runs for one
    resetting env: ``ScenarioRNG.reset_idx`` for the stream
    (hazard_nav_env.py:1564-1565) and the ``rotation_index`` lookup plus
    post-increment for the frozen layout (hazard_nav_env.py:2074-2080),
    including the LAZY creation of the counter array, which is why the first
    rung finds no array at all.
    """

    def __init__(self, num_envs=4, eval_seed=42, suite_s_class="single_row"):
        self.num_envs = num_envs
        self.device = "cpu"
        self.cfg = SimpleNamespace(
            seed=eval_seed,
            suite_s_class=suite_s_class,
            suite_s_index_rotation=True,
        )
        self._scenario = make_scenario_rng(self.cfg, num_envs, "cpu")
        self._layout_rng = np.random.default_rng(eval_seed)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)

    def finish_episode(self, env_index):
        """Reset one env; return (frozen layout index, protocol episode)."""
        self._scenario.reset_idx([env_index])
        if not self.cfg.suite_s_class:
            return None, int(self._scenario.episode_indices()[env_index])
        if not hasattr(self, "_suite_s_episode_counter"):
            self._suite_s_episode_counter = np.zeros(
                self.num_envs, dtype=np.int64
            )
        index = ROTATION_INDEX(
            env_index,
            int(self._suite_s_episode_counter[env_index]),
            rotation=bool(self.cfg.suite_s_index_rotation),
            layout_count=LAYOUT_COUNT,
        )
        self._suite_s_episode_counter[env_index] += 1
        return index, int(self._scenario.episode_indices()[env_index])


def run_rung(env, restart, episodes_per_env, mechanism="scenario-protocol"):
    """One rung: restart, then let every env finish some episodes.

    Returns ``(what the restart reported, {env: [(layout, episode), ...]})``.
    """
    reported = restart(env, mechanism)
    drawn = {index: [] for index in range(env.num_envs)}
    counts = (
        episodes_per_env
        if isinstance(episodes_per_env, dict)
        else {index: episodes_per_env for index in range(env.num_envs)}
    )
    for _ in range(max(counts.values())):
        for index in range(env.num_envs):
            if len(drawn[index]) < counts[index]:
                drawn[index].append(env.finish_episode(index))
    return reported, drawn


# ---------------------------------------------------------------- layer 3
# GAP 4: the Suite S rotation must be rewound with the protocol.


def test_restart_scenarios_rewinds_the_suite_s_rotation():
    """Three rungs, same frozen layouts, env for env and episode for episode."""
    namespace = load_sweep_namespace()
    restart = namespace["restart_scenarios"]
    env = FakeSuiteSEnv()
    _first, rung1 = run_rung(env, restart, 3)
    _second, rung2 = run_rung(env, restart, 3)
    _third, rung3 = run_rung(env, restart, 3)
    assert rung2 == rung1, (
        "rung 2 faced different Suite S layouts than rung 1: "
        f"{rung1} vs {rung2}"
    )
    assert rung3 == rung1, (
        "rung 3 faced different Suite S layouts than rung 1: "
        f"{rung1} vs {rung3}"
    )
    # And the layouts really are the rotation the env computes, not a constant
    # that would compare equal no matter what: env i's episode j is (i+j) % 10.
    expected = {
        index: [
            ((index + episode) % LAYOUT_COUNT, episode) for episode in range(3)
        ]
        for index in range(env.num_envs)
    }
    assert rung1 == expected, f"rotation is not the env's: {rung1}"
    note(f"3 rungs x {env.num_envs} envs x 3 episodes identical")


def test_the_fixture_would_notice_a_missing_rewind():
    """Teeth check: restart the PROTOCOL alone and the rungs come apart.

    This is the exact state of the world before the rotation rewind existed --
    restart_scenarios rebuilt the ScenarioRNG and nothing else -- so if this
    test ever passes, the test above proves nothing.
    """
    def protocol_only(env, _mechanism):
        env._scenario = make_scenario_rng(env.cfg, env.num_envs, env.device)
        return None

    env = FakeSuiteSEnv()
    _a, rung1 = run_rung(env, protocol_only, 3)
    _b, rung2 = run_rung(env, protocol_only, 3)
    assert rung2 != rung1, (
        "the fixture cannot see a missing rotation rewind, so the pairing "
        "test above is vacuous"
    )
    # Precisely: the protocol episode index is back to 0 while the frozen
    # layout has slid forward by the three episodes rung 1 ran.
    assert rung1[0][0] == (0, 0)
    assert rung2[0][0] == (3 % LAYOUT_COUNT, 0), rung2[0][0]
    note("protocol-only restart leaves the layout shifted by 3")


def test_suite_s_rewound_from_reports_the_drift():
    """The number written into the rung record is the drift that was removed."""
    namespace = load_sweep_namespace()
    restart = namespace["restart_scenarios"]
    env = FakeSuiteSEnv()
    first, _ = run_rung(env, restart, 3)
    second, _ = run_rung(env, restart, 5)
    third, _ = run_rung(env, restart, 1)
    # Rung 1 finds no counter at all (the env builds it lazily, at the first
    # reset), so it reports the start of the rotation.
    reported = (first, second, third)
    # Rung 2 rewinds a counter that rung 1 left at 3, rung 3 one left at 5.
    assert reported == (0, 3, 5), (
        f"restart_scenarios reported {reported} for three rungs of 3, 5 and 1 "
        "episodes per env; the rung records are supposed to carry the Suite S "
        "rotation drift each rung removed, and (0, 3, 5) is that drift. None "
        "in the tuple means the rotation was never rewound at all"
    )
    note("rewound_from = 0 / 3 / 5 across three rungs")


def test_unpaired_leaves_the_rotation_running():
    """--unpaired must reproduce the old behaviour, rotation included."""
    namespace = load_sweep_namespace(unpaired=True)
    restart = namespace["restart_scenarios"]
    unpaired = namespace["UNPAIRED"]
    assert namespace["pairing_mechanism"](FakeSuiteSEnv()) == unpaired
    env = FakeSuiteSEnv()
    reported, rung1 = run_rung(env, restart, 3, mechanism=unpaired)
    assert reported is None, (
        f"restart_scenarios reported a rewind of {reported} under {unpaired!r}"
    )
    _reported2, rung2 = run_rung(env, restart, 3, mechanism=unpaired)
    assert rung2 != rung1, (
        "--unpaired rewound the Suite S rotation; that flag exists to let the "
        "layout stream run on, and a curve taken with it would no longer be "
        "the unpaired control it claims to be"
    )
    assert env._suite_s_episode_counter.tolist() == [6] * env.num_envs
    note("rotation still advancing under --unpaired")


def test_non_suite_s_family_has_no_rotation_to_rewind():
    """A scatter id must report None, not 0, and must not grow a counter."""
    namespace = load_sweep_namespace()
    restart = namespace["restart_scenarios"]
    env = FakeSuiteSEnv(suite_s_class="")
    reported, _ = run_rung(env, restart, 3)
    assert reported is None, (
        f"a scatter id reported a Suite S rewind of {reported}; 0 would read "
        "in the curve JSON as a rotation that exists and is at its start"
    )
    assert not hasattr(env, "_suite_s_episode_counter")
    # And the scenario stream is still rewound for it.
    assert env._scenario.episode_indices().tolist() == [2] * env.num_envs
    reported2, _ = run_rung(env, restart, 1)
    assert reported2 is None, reported2
    assert env._scenario.episode_indices().tolist() == [0] * env.num_envs
    note("off Suite S: None reported, stream still rewound")


def test_ragged_rungs_still_pair():
    """Rungs of different lengths still start every env at the same layout.

    Two checkpoints end episodes at different rates, so the rungs consume
    different numbers of episodes per env.  The rotation is keyed on each env's
    OWN count, so the shared prefix has to match anyway -- which is the property
    that makes cpi_verdict's first-episode pairing legitimate.
    """
    namespace = load_sweep_namespace()
    restart = namespace["restart_scenarios"]
    env = FakeSuiteSEnv()
    _a, rung1 = run_rung(env, restart, {0: 4, 1: 1, 2: 3, 3: 2})
    _b, rung2 = run_rung(env, restart, {0: 1, 1: 3, 2: 2, 3: 4})
    for index in range(env.num_envs):
        shared = min(len(rung1[index]), len(rung2[index]))
        assert rung1[index][:shared] == rung2[index][:shared], (
            f"env {index}: {rung1[index]} vs {rung2[index]}"
        )
        assert shared >= 1
    note("ragged rungs share every episode they both ran")


# ---------------------------------------------------------------- layer 4
# GAP 5: the rung opening must not depend on what the skrl wrapper's reset does.


def _calls_in(node):
    """(receiver name or None, attribute or function name) for every call."""
    out = []
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Call):
            continue
        if isinstance(inner.func, ast.Attribute):
            receiver = getattr(inner.func.value, "id", None)
            out.append((receiver, inner.func.attr))
        elif isinstance(inner.func, ast.Name):
            out.append((None, inner.func.id))
    return out


def test_collect_does_not_reset_through_the_wrapper():
    """The rung boundary must be outside collect, and outside the wrapper."""
    resets = [
        call for call in _calls_in(_function_def("collect")) if call[1] == "reset"
    ]
    assert not resets, (
        f"collect() calls {resets}: the rung is being opened through a reset "
        "again.  If that reset is the skrl wrapper's, every rung after the "
        "first opens on an episode carried over from the previous checkpoint, "
        "and cpi_verdict.py pairs the rungs on exactly that record"
    )
    note("collect() opens no rung of its own")


def test_open_rung_resets_the_unwrapped_env():
    """open_rung must go to the env whose reset cannot be a no-op."""
    calls = _calls_in(_function_def("open_rung"))
    assert ("base", "reset") in calls, (
        f"open_rung does not call base.reset(): {calls}.  Only the unwrapped "
        "DirectRLEnv resets unconditionally; anything else leaves the rung "
        "opening at the mercy of the wrapper's once-only reset"
    )
    assert ("wrapped", "reset") not in calls, calls
    note("open_rung resets the unwrapped env")


def test_ladder_loop_opens_every_rung_before_collecting():
    """The per-checkpoint loop must open a rung and hand the obs to collect."""
    loops = [
        node
        for node in _script_tree().body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "ck"
    ]
    assert len(loops) == 1, f"found {len(loops)} ladder loops"
    body = loops[0]
    names = [call[1] for call in _calls_in(body)]
    for required in ("restart_scenarios", "open_rung", "rung_open_state",
                     "collect"):
        assert required in names, f"the ladder loop never calls {required}()"
    assert names.index("restart_scenarios") < names.index("open_rung"), (
        "the rewind must run BEFORE the reset that draws the rung's first "
        "episodes off the counters it puts back"
    )
    collects = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "collect"
    ]
    assert len(collects) == 1 and len(collects[0].args) == 2, (
        "collect() must be handed the opening observation, not fetch its own"
    )
    note("ladder loop: restart -> open -> collect(obs)")


def test_policy_observation_unwraps_the_policy_group():
    namespace = load_sweep_namespace()
    policy_observation = namespace["policy_observation"]
    tensor = torch.zeros((4, 42))
    assert policy_observation(({"policy": tensor}, {})) is tensor
    assert policy_observation({"policy": tensor}) is tensor
    assert policy_observation((tensor, {})) is tensor
    try:
        policy_observation(({"critic": tensor}, {}))
    except RuntimeError as exc:
        assert "policy" in str(exc)
    else:
        raise AssertionError("an observation dict with no policy group passed")
    note("policy group unwrapped, a missing one stops the sweep")


def test_check_observation_matches_rejects_a_transforming_wrapper():
    """A wrapper that reshapes observations must stop the sweep, not feed it."""
    namespace = load_sweep_namespace()
    check = namespace["check_observation_matches"]
    template = torch.zeros((4, 42))
    check(torch.zeros((4, 42)), template)
    check(torch.zeros((9, 9)), None)
    for wrong in (torch.zeros((4, 9)), torch.zeros((4, 42), dtype=torch.float64),
                  np.zeros((4, 42), dtype=np.float32)):
        try:
            check(wrong, template)
        except RuntimeError:
            continue
        raise AssertionError(f"accepted an observation of {type(wrong)} {wrong.shape}")
    note("shape / dtype / type mismatches all stop the sweep")


def test_rung_open_state_reports_a_carry_over():
    """The diagnostic has to be able to SAY a rung opened mid-episode."""
    namespace = load_sweep_namespace()
    namespace["base"] = FakeSuiteSEnv()
    rung_open_state = namespace["rung_open_state"]
    # Fresh: nothing mid-episode, counters at their first episode.
    for index in range(namespace["base"].num_envs):
        namespace["base"].finish_episode(index)
    fresh, span = rung_open_state()
    assert fresh is True, fresh
    assert span == [0, 0], span
    # Carried over: one env is 37 steps into an episode and two episodes ahead.
    namespace["base"].episode_length_buf[2] = 37
    namespace["base"].finish_episode(2)
    fresh, span = rung_open_state()
    assert fresh is False, fresh
    assert span == [0, 1], span
    # A family carrying neither must not crash the sweep, it must say None.
    namespace["base"] = SimpleNamespace()
    assert rung_open_state() == (None, None)
    note("opened_fresh / opened_episode_index report both states")


def test_curve_json_carries_the_pairing_diagnostics():
    """A reader must be able to audit both mechanisms from the JSON alone."""
    dicts = [node for node in ast.walk(_script_tree()) if isinstance(node, ast.Dict)]
    keysets = [
        {
            key.value
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        for node in dicts
    ]
    headers = [keys for keys in keysets if "rungs" in keys]
    assert headers, "no curve header dict"
    for field in ("pairing", "suite_s_class", "suite_s_rotation", "rung_open",
                  "wrapper_reset_one_shot"):
        assert any(field in keys for keys in headers), (
            f"the curve header dropped {field!r}, so a reader cannot tell "
            "whether this sweep's rungs were actually paired"
        )
    rungs = [keys for keys in keysets if "checkpoint" in keys]
    assert rungs, "no rung record dict"
    for field in ("suite_s_rewound_from", "opened_fresh",
                  "opened_episode_index"):
        assert any(field in keys for keys in rungs), (
            f"the rung record dropped {field!r}, which is the per-rung "
            "evidence for the header's pairing claim"
        )
    note("header + rung records carry the audit fields")


# ---------------------------------------------------------------- runner


def main() -> int:
    failures = 0
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    for name, fn in tests:
        NOTES.clear()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - report, do not abort the run
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            detail = "; ".join(NOTES)
            print(f"PASS {name}" + (f": {detail}" if detail else ""))
    print(f"test_convergence_curve_pairing: {len(tests)} tests, "
          f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
