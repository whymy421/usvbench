"""G35 (controller decision J19, v2.6.1): the PPO trainer survives a failed heartbeat replace and counts appearing
triggers outside the speed envelope instead of dying on them. CPU only; nothing is trained (the G19b fake backend
drives the REAL train_journal_ppo.run_training).

Covers:
  * retrying_atomic_write_json: retries on OSError up to HEARTBEAT_REPLACE_ATTEMPTS times, HEARTBEAT_REPLACE_WAIT_S
    apart, returns the last error instead of raising; succeeds as soon as one attempt works;
  * a whole fake training run whose heartbeat replace ALWAYS fails (PermissionError, as when a Windows reader holds
    heartbeat.json) completes, writes a valid training record and counts the skipped heartbeats; every other file
    (preflight.json, the record) is written normally;
  * StepMonitor counts finished episodes whose appearing stamp carries trigger_speed_outside_envelope (and lists the
    first ones) in divergence_checks.appearing_trigger_speed; the flag names equal the stamp module's.

Run: python -B -m pytest -p no:cacheprovider scripts/test_g35_trainer_resilience.py -q
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


G = _load("_g35_g19b_tests", SCRIPTS / "test_g19b_train_journal_ppo.py")     # fake backend, plan_for
T, R = G.T, G.R
ST = _load("_g35_appearing_stamp", SCRIPTS.parent / "tasks" / "_shared" / "conditions" / "appearing" / "stamp.py")


def scratch_dir():
    """A fresh folder under <repo>/../test_scratch (kept; this test deletes nothing)."""
    root = SCRIPTS.parents[1] / "test_scratch"
    root.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="g35-train-", dir=str(root)))


def test_flag_names_equal_the_stamp_module():
    assert T.APPEARING_SPEED_FLAG == ST.SPEED_ENVELOPE_FLAG
    assert T.APPEARING_SPEED_RECORD == ST.SPEED_ENVELOPE_RECORD


def test_retrying_write_retries_then_gives_up_without_raising():
    calls, sleeps = [], []

    def flaky(fail_times):
        def write(path, value):
            calls.append(path)
            if len(calls) <= fail_times:
                raise PermissionError(13, "Access is denied", str(path))
        return write

    assert T.retrying_atomic_write_json("hb", {}, sleep=sleeps.append, write=flaky(2)) is None
    assert len(calls) == 3 and sleeps == [T.HEARTBEAT_REPLACE_WAIT_S] * 2
    calls.clear(), sleeps.clear()
    error = T.retrying_atomic_write_json("hb", {}, sleep=sleeps.append, write=flaky(99))
    assert isinstance(error, PermissionError)
    assert len(calls) == T.HEARTBEAT_REPLACE_ATTEMPTS and len(sleeps) == T.HEARTBEAT_REPLACE_ATTEMPTS - 1
    calls.clear()

    def broken(path, value):
        raise ValueError("not an OSError")
    try:
        T.retrying_atomic_write_json("hb", {}, sleep=sleeps.append, write=broken)
    except ValueError:
        pass
    else:
        raise AssertionError("only OSError is retried and skipped")


def test_training_completes_when_every_heartbeat_replace_fails(monkeypatch):
    out = scratch_dir() / "run"
    plan, args = G.plan_for(out)
    original = T.atomic_write_json

    def failing_heartbeat(path, value):
        if Path(path).name == "heartbeat.json":
            raise PermissionError(13, "Access is denied", str(path))
        return original(path, value)

    monkeypatch.setattr(T, "atomic_write_json", failing_heartbeat)
    monkeypatch.setattr(T.time, "sleep", lambda s: None)
    assert T.run_training(plan, args, G.fake_backend(4)) == T.EXIT_OK
    record = R.load_record(out / "training_record.json", run_dir=out)
    assert record["status"] == "completed" and record["completed_timesteps"] == 640
    skipped = record["divergence_checks"]["heartbeats_skipped"]
    assert skipped == 640 // plan["rollouts"] + 2, skipped                  # starting + every iteration + finishing
    assert not (out / "heartbeat.json").exists() and "heartbeat.json" not in record["log_digests"]
    assert record["divergence_checks"]["appearing_trigger_speed"]["outside_envelope"] == 0


def test_monitor_counts_out_of_envelope_appearing_triggers():
    flagged = {"payload": {"runtime": {ST.SPEED_ENVELOPE_FLAG: True, ST.SPEED_ENVELOPE_RECORD: {
        "speed_mps": -1.5084264743885187, "envelope_mps": [-0.9743162233660108, 3.000326502719915], "rule": "x"}}}}
    normal = {"payload": {"runtime": {"exposure_realized": True}}}
    base = types.SimpleNamespace(episode_appearing_hazards=[flagged, normal, None])
    wrapped = types.SimpleNamespace(step=lambda actions: None, num_envs=3)
    plan = {"budget_timesteps": 1000, "rollouts": 10, "num_envs": 3, "code9_reward": 0.0, "code9_source": None}
    monitor = T.StepMonitor(wrapped, base, plan, scratch_dir(), lambda: None)
    try:
        monitor.components = ["appearing", "calm", "calm"]
        monitor._count_appearing([True, True, True])
        monitor._count_appearing([False, True, False])
        block = monitor.divergence_checks()["appearing_trigger_speed"]
        assert (block["stamps_seen"], block["outside_envelope"]) == (3, 1)
        assert block["examples"] == [{"timestep": 0, "env": 0, "component": "appearing",
                                      "speed_mps": -1.5084264743885187,
                                      "envelope_mps": [-0.9743162233660108, 3.000326502719915]}]
        assert block["rule"] == T.APPEARING_SPEED_RULE
        json.dumps(monitor.divergence_checks(), allow_nan=False)
        no_stamps = T.StepMonitor(types.SimpleNamespace(step=lambda a: None, num_envs=3), types.SimpleNamespace(),
                                  plan, scratch_dir(), lambda: None)
        no_stamps._count_appearing([True, True, True])                       # a family without appearing stamps
        assert no_stamps.divergence_checks()["appearing_trigger_speed"]["stamps_seen"] == 0
        no_stamps.close()
    finally:
        monitor.close()
