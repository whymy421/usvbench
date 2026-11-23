"""Early stop on a long stretch without a successful training episode (scripts/ppo_early_stop.py,
usvbench_ppo_fix_runs/STOP_RULE_BACKTEST.md). CPU only; the G19b fake backend drives the REAL
train_journal_ppo.run_training, nothing is trained.

Covers:
  * NoSuccessStop: no stop before min_step; a stop exactly when the window holds no success; a success moves the
    window; a resumed run restarts the window; no termination codes = never stops;
  * a fake run that never succeeds stops at the first rollout boundary at or after 1/3 of the budget, writes a valid
    record with status early_stopped, final.pt = the latest saved checkpoint and early_stop.json; exit code OK;
  * a fake run whose row 0 succeeds every episode completes normally (early_stop.json says it never stopped);
  * without the flag the plan carries early_stop = None and nothing changes.

Run: python -B -m pytest -p no:cacheprovider scripts/test_ppo_early_stop.py -q
"""
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


G = _load("_es_g19b_tests", SCRIPTS / "test_g19b_train_journal_ppo.py")      # fake backend, args_for
T, R = G.T, G.R
import ppo_early_stop as ES  # noqa: E402


def scratch_dir():
    root = SCRIPTS.parents[1] / "test_scratch"
    root.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="es-train-", dir=str(root)))


def plan_with(out, early_stop=True, budget=640):
    extra = ["--early-stop-no-success"] if early_stop else []
    args = G.args_for(out, "--arm", "mixture", "--budget-timesteps", str(budget), "--num-envs", "4",
                      "--checkpoint-every", str(budget // 10), "--device", "cpu", *extra, task="journal:reach")
    plan = T.preflight(args, root=T.namespace_tasks(), reference_block=G.NO_MATCH, code_tree_sha256=G.tree_sha())
    return plan, args


# ----------------------------------------------------------------------------------------------- the rule itself
def test_rule_unit():
    s = ES.NoSuccessStop(ES.DEFAULTS, 300)
    assert (s.min_step, s.window) == (100, 100)
    s.observe(10, [True, False], [3, 0])
    assert s.check(99) is None                              # before min_step
    block = s.check(100)
    assert block is not None and block["last_success_timestep"] is None and block["window_start"] == 0
    s = ES.NoSuccessStop(ES.DEFAULTS, 300)
    s.observe(50, [False, True], [0, 1])                    # a success at 50
    assert s.check(100) is None and s.check(149) is None
    assert s.check(150)["window_start"] == 50
    s = ES.NoSuccessStop(ES.DEFAULTS, 300)
    s.restart(120)                                          # resumed at 120: the window starts again
    assert s.check(200) is None and s.check(220) is not None
    s = ES.NoSuccessStop(ES.DEFAULTS, 300)
    s.observe(5, [True], None)                              # no codes published: never stops
    assert s.check(299) is None and s.summary()["codes_missing_steps"] == 1
    s = ES.NoSuccessStop(ES.DEFAULTS, 300)
    s.observe(5, [False, False], None)                      # no row done: nothing to count, not "missing"
    assert s.summary()["codes_missing_steps"] == 0


# ----------------------------------------------------------------------------------------------- real trainer
def test_never_succeeding_run_stops_early_with_a_valid_record():
    out = scratch_dir() / "run"
    plan, args = plan_with(out)
    assert plan["early_stop"] == ES.DEFAULTS
    assert T.run_training(plan, args, G.fake_backend(4)) == T.EXIT_OK
    record = R.load_record(out / "training_record.json", run_dir=out)
    rollouts = plan["rollouts"]
    first = int(math.ceil(640 / 3.0 / rollouts)) * rollouts      # first rollout boundary >= ceil(640/3)
    assert record["status"] == "early_stopped" and record["completed_timesteps"] == first, record["completed_timesteps"]
    assert record["divergence"] is None
    latest = max(c["timestep"] for c in record["checkpoints"])
    assert latest <= first and record["final_checkpoint"]["source_path"].endswith("agent_%d.pt" % latest)
    assert R.file_sha256(out / "final.pt") == record["final_checkpoint"]["sha256"]
    evidence = json.loads((out / "early_stop.json").read_text(encoding="utf-8"))
    assert evidence["stopped"]["timestep"] == first and evidence["successes_seen"] == 0
    assert evidence["episodes_seen"] > 0
    assert "early_stop.json" in record["log_digests"]


def test_succeeding_run_completes(monkeypatch):
    original = G.FakeEnv.step

    def step(self, actions):
        out = original(self, actions)
        done = out[2]
        if bool(done[0, 0]) and int(self.termination_reason[0]) != 9:
            self.termination_reason[0] = 1                  # row 0 ends every episode with success
        return out

    monkeypatch.setattr(G.FakeEnv, "step", step)
    out = scratch_dir() / "run"
    plan, args = plan_with(out)
    assert T.run_training(plan, args, G.fake_backend(4)) == T.EXIT_OK
    record = R.load_record(out / "training_record.json", run_dir=out)
    assert record["status"] == "completed" and record["completed_timesteps"] == 640
    evidence = json.loads((out / "early_stop.json").read_text(encoding="utf-8"))
    assert evidence["stopped"] is None and evidence["successes_seen"] > 0


def test_flag_off_changes_nothing():
    out = scratch_dir() / "run"
    plan, args = plan_with(out, early_stop=False)
    assert plan["early_stop"] is None
    assert T.run_training(plan, args, G.fake_backend(4)) == T.EXIT_OK
    record = R.load_record(out / "training_record.json", run_dir=out)
    assert record["status"] == "completed" and not (out / "early_stop.json").exists()
