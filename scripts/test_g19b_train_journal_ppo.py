"""G19b (IMPLEMENTATION_PLAN P5 / P6 / P4): journal PPO training entry point, training record, evaluator refusals.

CPU only; nothing is trained. The run path is driven by a FAKE backend (a stub env + a stub skrl runner that writes
agent_<t>.pt files the way skrl names them), so the entry point's own logic is exercised for real: preflight,
stamped agent overrides, heartbeat / progress, divergence and failure statuses, final-checkpoint rule, record
written last and validated.

Run: python -B -m pytest -p no:cacheprovider scripts/test_g19b_train_journal_ppo.py -q
"""
import contextlib
import copy
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types

import pytest

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


T = load("train_journal_ppo")
R = sys.modules["ppo_training_record"]
SR = sys.modules["seed_registry"]
EP = sys.modules["eval_protocol"]
import torch  # noqa: E402

SMOKE = 90001
MATCH = {"id": "RE-1", "versions": {}, "match": True, "mismatches": []}
NO_MATCH = {"id": "RE-1", "versions": {}, "match": False, "mismatches": ["python=3.13 (required 3.11.15)"]}
TREE = "e" * 64
_TREE_SHA = {}


def tree_sha():
    if "sha" not in _TREE_SHA:
        _TREE_SHA["sha"] = T.code_tree.manifest_from_tree(T.REPO)[2]
    return _TREE_SHA["sha"]


@pytest.fixture
def scratch():
    root = SCRIPTS.parents[1] / "test_scratch"
    root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="g19b-train-", dir=root) as directory:
        yield Path(directory)


def args_for(out, *extra, task="journal:reach", vessel="blueboat", seed=SMOKE):
    return T.build_parser().parse_args(["--task", task, "--vessel", vessel, "--training-seed", str(seed),
                                        "--out", str(out)] + list(extra))


def run_main(argv, backend=None):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = T.main(argv, backend=backend)
    return code, out.getvalue(), err.getvalue()


def frozen_registry():
    registry = SR.load_registry()
    registry["status"], registry["competence_gate"]["status"] = "frozen", "frozen"
    registry["g52_test_note"] = "another frozen registry (the in-tree file is frozen since v2.6.6)"
    return SR.validate_registry(registry)


def nested_diff(a, b, prefix=""):
    """Dotted paths whose values differ between two nested dicts."""
    out = set()
    for key in set(a) | set(b):
        path = prefix + str(key)
        if isinstance(a.get(key), dict) and isinstance(b.get(key), dict):
            out |= nested_diff(a[key], b[key], path + ".")
        elif a.get(key) != b.get(key) or (key in a) != (key in b):
            out.add(path)
    return out


# ------------------------------------------------------------------ dry run / preflight

def test_dry_run_preflight_for_all_24_journal_ids():
    plans = EP.journal_plans()
    ids = sorted(plans.JOURNAL_ENV_IDS.items())
    assert len(ids) == 24
    seen_appearing = 0
    for (task, vehicle), gym_id in ids:
        vessel = "wamv" if vehicle.startswith("wamv") else "blueboat"
        for arm in ("mixture", "calm"):
            if arm == "calm" and task not in ("scatter", "path_following"):
                continue
            plan = T.preflight(args_for(Path("unused-dir-g19b") / gym_id, "--dry-run", "--arm", arm, task=gym_id,
                                        vessel=vessel), reference_block=NO_MATCH, code_tree_sha256=TREE)
            assert plan["journal_id"] == gym_id and plan["task"] == task and plan["vehicle"] == vehicle
            assert plan["purpose"] == "training_smoke" and plan["registry"]["block_id"] == "smoke-90001"
            assert plan["budget_timesteps"] == T.DEFAULT_BUDGET[vessel]
            assert plan["checkpoint_every"] * 50 == plan["budget_timesteps"]      # v2.6.4: every 2 % (G47)
            assert [o["path"] for o in plan["agent"]["overrides"]] == list(R.ALLOWED_AGENT_OVERRIDES)
            assert plan["training_pool_module"] == plan["carrier"] + ".training_pool"
            if arm == "mixture":
                assert plan["appearing_wired"] == (task in T.task_modules(T.namespace_tasks()).appearing.WIRED_TASKS)
                # G31 / owner decision J10: training dose = D1, except BlueBoat station keeping (D2: no D1 by design)
                mix = T.task_modules(T.namespace_tasks()).mixture
                dl, al = mix.training_level(task, vehicle, "dynamic"), mix.training_level(task, vehicle, "appearing")
                assert (dl, al) == ((2, 2) if (task, vehicle) == ("station_keeping", "blueboat_v2") else (1, 1))
                assert ("dynamic=D%d" % dl) in plan["stack_tokens"] and "waves=D1" in plan["stack_tokens"]
                assert (("appearing=D%d" % al) in plan["stack_tokens"]) == plan["appearing_wired"]
                assert plan["mixture"]["calm_slots"] == 3
                seen_appearing += plan["appearing_wired"]
            else:
                assert plan["mixture"] is None and plan["stack_tokens"] == []
            json.dumps(plan, allow_nan=False)
    assert seen_appearing >= 4                      # v2.5: scatter and path following, both vessels


def test_cli_dry_run_writes_nothing_but_the_requested_preflight(scratch):
    out = scratch / "run"
    target = scratch / "preflight.json"
    code, text, err = run_main(["--task", "Isaac-USV-Dock-BlueBoat-Journal-Direct-v1", "--vessel", "blueboat",
                                "--training-seed", str(SMOKE), "--out", str(out), "--dry-run",
                                "--preflight-out", str(target)])
    assert code == 0, err
    assert not out.exists() and target.is_file()
    plan = json.loads(target.read_text(encoding="ascii"))
    assert plan["agent"]["entry_point"] == "docking.agents:skrl_ppo_blueboat_cfg.yaml"      # P9
    assert "DRY RUN OK" in text


@pytest.mark.parametrize("extra,seed,task,match", [
    ((), 11001, "journal:reach", "selection"),         # evaluation seeds are refused (every purpose)
    ((), 21001, "journal:reach", "pilot"),
    ((), 31001, "journal:reach", "calibration"),
    ((), 41041, "journal:reach", "final_test"),
    ((), 42, "journal:reach", "release-record"),       # G52: campaign seed, frozen registry: needs the release
    ((), 55555, "journal:reach", "not a training seed"),
    (("--purpose", "training"), SMOKE, "journal:reach", "training_smoke"),
    ((), SMOKE, "Isaac-USV-HazardNav-Direct-v1", "only journal ids"),
    ((), SMOKE, "journal:nope", "unknown journal task"),
    (("--agent-set", "agent.learning_rate=1e-3"), SMOKE, "journal:reach", "agent-set"),
    (("--budget-timesteps", "96001"), SMOKE, "journal:reach", "divide"),
    (("--budget-timesteps", "1000", "--checkpoint-every", "100"), SMOKE, "journal:reach", "iterations"),
])
def test_preflight_refusals(scratch, extra, seed, task, match):
    code, _, err = run_main(["--task", task, "--vessel", "blueboat", "--training-seed", str(seed),
                             "--out", str(scratch / "run"), "--dry-run"] + list(extra))
    assert code == T.EXIT_REFUSED and match in err, err


def test_existing_out_dir_and_unknown_arguments_are_refused(scratch):
    (scratch / "run").mkdir()
    code, _, err = run_main(["--task", "journal:reach", "--vessel", "blueboat", "--training-seed", str(SMOKE),
                             "--out", str(scratch / "run"), "--dry-run"])
    assert code == T.EXIT_REFUSED and "exists" in err
    code, _, err = run_main(["--task", "journal:reach", "--vessel", "blueboat", "--training-seed", str(SMOKE),
                             "--out", str(scratch / "new"), "--dry-run", "env.seed=3"])
    assert code == T.EXIT_REFUSED and "unknown arguments" in err
    with pytest.raises(T.TrainingRefusal):
        T.isaac_backend(["env.scene.num_envs=8"], "cuda:0")


def test_campaign_seed_needs_frozen_in_tree_registry_release_and_re1(scratch, monkeypatch):
    registry = frozen_registry()
    args = args_for(scratch / "run", seed=1001)
    # follow-up M4: a frozen registry that is not the in-tree file is refused for campaign seeds
    with pytest.raises(T.TrainingRefusal, match="in-tree seed registry"):
        T.preflight(args, registry=registry, reference_block=MATCH, code_tree_sha256=TREE)
    forged = scratch / "forged_registry.json"
    forged.write_text(json.dumps(registry, indent=2), encoding="ascii")
    code, _, err = run_main(["--task", "journal:reach", "--vessel", "blueboat", "--training-seed", "1001",
                             "--out", str(scratch / "run"), "--dry-run", "--seed-registry", str(forged)])
    assert code == T.EXIT_REFUSED and "in-tree seed registry" in err
    in_tree = scratch / "in_tree_frozen.json"                  # the owner's frozen in-tree file
    in_tree.write_text(json.dumps(registry, indent=2), encoding="ascii")
    monkeypatch.setattr(T, "IN_TREE_REGISTRY", in_tree)
    with pytest.raises(T.TrainingRefusal, match="release-record"):
        T.preflight(args, registry=registry, reference_block=MATCH, code_tree_sha256=TREE)
    release = scratch / "release.json"
    release.write_text(json.dumps({"release_id": "REL1-test", "code_tree_sha256": "f" * 64}), encoding="ascii")
    args = args_for(scratch / "run", "--release-record", str(release), seed=1001)
    with pytest.raises(T.TrainingRefusal, match="RE-1"):
        T.preflight(args, registry=registry, reference_block=NO_MATCH, code_tree_sha256=TREE)
    with pytest.raises(T.TrainingRefusal, match="pins code tree"):
        T.preflight(args, registry=registry, reference_block=MATCH, code_tree_sha256=TREE)
    release.write_text(json.dumps({"release_id": "REL1-test", "code_tree_sha256": TREE}), encoding="ascii")
    plan = T.preflight(args, registry=registry, reference_block=MATCH, code_tree_sha256=TREE)
    assert plan["purpose"] == "training" and plan["registry"]["candidate_index"] == 1
    assert plan["release"]["release_id"] == "REL1-test" and plan["re1_match"] is True
    assert plan["release"]["code_tree_sha256"] == TREE


def test_app_arguments_pass_the_device_and_refuse_hydra():
    """Follow-up m6: AppLauncher gets --device (unless the AppLauncher arguments name one)."""
    assert T.app_arguments(["--headless"], "cuda:1") == ["--headless", "--device", "cuda:1"]
    assert T.app_arguments(["--device", "cpu"], "cuda:0") == ["--device", "cpu"]
    for bad in (["env.scene.num_envs=8"], ["agent.seed=3"], ["+x=1"]):
        with pytest.raises(T.TrainingRefusal):
            T.app_arguments(bad, "cuda:0")


def test_finish_follows_the_evaluator_exit_pattern():
    """Follow-up M1: non-zero -> os._exit(code) without close(); 0 -> close() under a watchdog, then exit 0."""
    calls = []

    class App:
        def close(self):
            calls.append("close")

    class Timer:
        def __init__(self, seconds, function):
            calls.append(("watchdog", seconds))
            self.daemon = False

        def start(self):
            calls.append("watchdog-started")

    backend = types.SimpleNamespace(app=App())
    for code in (2, 3, 4):
        del calls[:]
        T.finish(code, backend, exit_fn=lambda c: calls.append(("exit", c)), timer_cls=Timer)
        assert calls == [("exit", code)]
    del calls[:]
    T.finish(0, backend, exit_fn=lambda c: calls.append(("exit", c)), timer_cls=Timer)
    assert calls == [("watchdog", T.CLOSE_WATCHDOG_S), "watchdog-started", "close", ("exit", 0)]
    assert T.finish(3, types.SimpleNamespace()) == 3                   # no app (CPU tests): plain return


def test_agent_override_set_equals_the_stamped_set():
    plan = T.preflight(args_for(Path("unused-g19b"), "--dry-run"), reference_block=NO_MATCH, code_tree_sha256=TREE)
    base, _ = T.load_agent_yaml(plan["agent"]["entry_point"])
    applied = copy.deepcopy(base)
    stamped = T.agent_overrides(applied, seed=SMOKE, budget=plan["budget_timesteps"], directory="d",
                                experiment_name="x", checkpoint_interval=plan["checkpoint_every"], write_interval=960)
    assert [s["path"] for s in stamped] == list(R.ALLOWED_AGENT_OVERRIDES)
    assert nested_diff(base, applied) <= set(R.ALLOWED_AGENT_OVERRIDES)
    for item in stamped:
        assert R.get_path(applied, item["path"]) == item["value"]
    assert R.get_path(applied, "agent.experiment.wandb") is False


def test_appearing_wiring_is_read_at_run_time():
    """G20b wires the appearing suite into more tasks by extending conditions/appearing/config.WIRED_TASKS (and the
    family's appearing_admission_inputs): the mixture, the preflight and the shared adapter follow without a change."""
    mods = T.task_modules(T.namespace_tasks())
    TH = mods.hazards
    saved = mods.appearing.WIRED_TASKS
    try:
        mods.appearing.WIRED_TASKS = tuple(t for t in saved if t != "path_hazard")    # v2.5 state (G20b wires it)
        before = T.preflight(args_for(Path("unused-g19b"), "--dry-run", task="journal:path_hazard"),
                             reference_block=NO_MATCH, code_tree_sha256=TREE)
        assert before["appearing_wired"] is False and "appearing=D1" not in before["stack_tokens"]
        mods.appearing.WIRED_TASKS = tuple(t for t in saved if t != "path_hazard") + ("path_hazard",)
        after = T.preflight(args_for(Path("unused-g19b"), "--dry-run", task="journal:path_hazard"),
                            reference_block=NO_MATCH, code_tree_sha256=TREE)
        assert after["appearing_wired"] is True and "appearing=D1" in after["stack_tokens"]
        assert after["mixture"]["block_length"] == 10 and after["mixture"]["weights"]["calm"] == "3/10"
        mixture = mods.mixture.TrainingMixture(seed=SMOKE, task="path_hazard", vessel="blueboat_v2",
                                               appearing_wired=True, purpose="training_smoke")
        common = dict(mixture=mixture, task="path_hazard", vessel="blueboat_v2", seed=SMOKE, tier=2, num_envs=2,
                      cache=TH.TrainingHazardCache(), purpose="training_smoke")
        wired = types.SimpleNamespace(__name__="fake_family_journal",
                                      appearing_admission_inputs=lambda vessel, tier, seed, env, episode: None)
        pool = TH.appearing_pool_for((types.SimpleNamespace(__name__="other"), wired), **common)
        assert isinstance(pool, TH.TrainingAppearingPool) and pool.task == "path_hazard"
        with pytest.raises(TH.TrainingHazardError, match="appearing_admission_inputs"):
            TH.appearing_pool_for(types.SimpleNamespace(__name__="unwired_family"), **common)
        calls = []

        def six(task, vessel, tier, seed, env, episode):
            calls.append((task, vessel, tier, seed, env, episode))
        TH.call_admission_inputs(six, "path_hazard", "blueboat_v2", 2, SMOKE, 1, 3)
        TH.call_admission_inputs(lambda vessel, tier, seed, env, episode: calls.append((vessel, env, episode)),
                                 "path_hazard", "blueboat_v2", 2, SMOKE, 1, 3)
        assert calls == [("path_hazard", "blueboat_v2", 2, SMOKE, 1, 3), ("blueboat_v2", 1, 3)]
    finally:
        mods.appearing.WIRED_TASKS = saved


# ------------------------------------------------------------------ fake backend

class FakeLayer:
    def __init__(self, mixture, n):
        self.training_mixture = mixture
        self.episode = [0] * n
        self.counts = {}
        for i in range(n):
            self._count(i)

    def _count(self, i):
        name = self.training_component(i)
        self.counts[name] = self.counts.get(name, 0) + 1

    def training_component(self, i):
        return "calm" if self.training_mixture is None else self.training_mixture.component(i, self.episode[i])

    def training_mixture_counts(self):
        return dict(self.counts)


class FakeEnv:
    """nan_at: a NaN reward on row 0 that is NOT a code-9 termination; code9_at: row 0 terminates with code 9 and
    a NaN reward (the legal non-finite-state outcome); nan_obs_at: a NaN observation on row 1."""

    def __init__(self, cfg, n, nan_at=None, code9_at=None, nan_obs_at=None, detach=False, done_period=7):
        self.cfg = cfg
        self.num_envs = n
        self.unwrapped = self
        self._conditions = FakeLayer(None if detach else getattr(cfg, "journal_training_mixture", None), n)
        self._d7_pool = getattr(cfg, "journal_d7_pool", None)
        self._appearing_pool = getattr(cfg, "journal_appearing_pool", None)
        self.termination_reason = torch.zeros(n, dtype=torch.long)
        self.t = 0
        self.nan_at, self.nan_obs_at, self.done_period = nan_at, nan_obs_at, done_period
        self.code9_steps = set(code9_at) if isinstance(code9_at, (tuple, list, set)) else (
            set() if code9_at is None else {code9_at})
        box = torch.tensor([[-10.0, 40.0, -22.0, 22.0]] * n, dtype=torch.float64)   # G44: layout bounds
        for name in ("_journal_bounds", "_jph_bounds", "_jbounds", "_jd_bounds"):
            setattr(self, name, box.clone())

    def step(self, actions):
        self.t += 1
        n = self.num_envs
        rew = torch.full((n, 1), 0.1 * (self.t % 5))
        done = torch.tensor([[(self.t + i) % self.done_period == 0] for i in range(n)])
        codes = torch.zeros(n, dtype=torch.long)
        if self.nan_at is not None and self.t == self.nan_at:
            rew[0, 0] = float("nan")
            done[0, 0] = False
        if self.t in self.code9_steps:
            rew[0, 0] = float("nan")
            done[0, 0] = True
        obs = torch.zeros((n, 4))
        if self.nan_obs_at is not None and self.t == self.nan_obs_at:
            obs[1, 2] = float("nan")
        for i in range(n):
            if bool(done[i, 0]):
                codes[i] = 9 if (self.t in self.code9_steps and i == 0) else (
                    5 if i % 2 else 3)
                self._conditions.episode[i] += 1
                self._conditions._count(i)
        self.termination_reason = codes
        return obs, rew, done, torch.zeros_like(done), {}


class FakeScaler(torch.nn.Module):
    """G44: skrl RunningStandardScaler stand-in (forward(x, train=False, inverse=False, no_grad=True))."""

    def __init__(self):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(4, dtype=torch.float64))
        self.register_buffer("current_count", torch.ones((), dtype=torch.float64))

    def forward(self, x, train=False, inverse=False, no_grad=True):
        if train:
            self.current_count += 1.0
            self.running_mean += (x.double().mean(0) - self.running_mean) / self.current_count
        return x


class FakeAgent:
    """skrl-like agent: _update / track_data / checkpoint_modules (a model, an optimizer); G44: a state
    preprocessor and a KLAdaptiveLR-like scheduler (min_lr) whose learning rate halves at every update."""

    def __init__(self):
        self.model = torch.nn.Linear(2, 2)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self._state_preprocessor = FakeScaler()
        self.scheduler = types.SimpleNamespace(min_lr=1.0e-6)
        self.checkpoint_modules = {"policy": self.model, "optimizer": self.optimizer,
                                   "state_preprocessor": self._state_preprocessor}
        self.tracked = []

    def _update(self, timestep, timesteps):
        self._state_preprocessor(torch.full((3, 4), float(timestep)), train=True)     # skrl: first epoch
        group = self.optimizer.param_groups[0]
        group["lr"] = max(group["lr"] / 2.0, self.scheduler.min_lr)
        return None

    def track_data(self, tag, value):
        self.tracked.append((tag, value))


class FakeRunner:
    def __init__(self, wrapped, cfg, fail_at=None, bad_final=False, skip_final=False, nan_loss_at=None,
                 nan_params_at=None, raise_after_nan_params=False, nan_action_at=None, raise_on_nan_obs=False):
        self.wrapped, self.cfg = wrapped, cfg
        self.agent = FakeAgent()
        self.fail_at, self.bad_final, self.skip_final = fail_at, bad_final, skip_final
        self.nan_loss_at, self.nan_params_at, self.raise_after = nan_loss_at, nan_params_at, raise_after_nan_params
        self.nan_action_at, self.raise_on_nan_obs = nan_action_at, raise_on_nan_obs
        self.rewards_seen = []

    def run(self, mode):
        assert mode == "train"
        total = self.cfg["trainer"]["timesteps"]
        rollouts = self.cfg["agent"]["rollouts"]
        exp = self.cfg["agent"]["experiment"]
        assert self.cfg["trainer"]["close_environment_at_exit"] is False            # follow-up M1
        folder = Path(exp["directory"]) / exp["experiment_name"]
        (folder / "checkpoints").mkdir(parents=True)
        (folder / "events.out.tfevents.0.fake").write_bytes(b"fake event file")
        for t in range(total):
            if self.fail_at is not None and t == self.fail_at:
                raise RuntimeError("simulated host failure")
            actions = torch.zeros((self.wrapped.num_envs, 2))
            if self.nan_action_at is not None and t == self.nan_action_at:
                actions[1, 0] = float("nan")
            out = self.wrapped.step(actions)
            self.rewards_seen.append(out[1].clone())
            if self.raise_on_nan_obs and not bool(torch.isfinite(out[0]).all()):
                # what skrl's act() does with a NaN observation: torch.distributions raises a ValueError
                raise ValueError("Expected parameter loc (Tensor of shape (4, 2)) of distribution Normal to satisfy "
                                 "the constraint Real(), but found invalid values")
            step = t + 1
            if step % rollouts == 0:
                if self.nan_params_at is not None and step >= self.nan_params_at:
                    with torch.no_grad():
                        self.agent.model.weight[0, 0] = float("nan")
                    if self.raise_after:      # what skrl does at the next act(): torch.distributions ValueError
                        raise ValueError("Expected parameter loc (Tensor of shape (1,)) of distribution Normal to "
                                         "satisfy the constraint Real(), but found invalid values")
                self.agent._update(t, total)
                loss = float("nan") if (self.nan_loss_at is not None and step >= self.nan_loss_at) else 0.5
                self.agent.track_data("Loss / Policy loss", loss)
            if step % exp["checkpoint_interval"] == 0 and not (self.skip_final and step == total):
                value = float("nan") if (self.bad_final and step == total) else float(step)
                torch.save({"policy": {"w": torch.full((3,), value)}, "step": step},
                           folder / "checkpoints" / ("agent_%d.pt" % step))


G44_FAKE_CFG = {"reward_progress_scale": 20.0, "reward_clearance_scale": 0.0, "reward_prox_scale": 4.1,
                "safe_clearance_m": 0.9, "half_beam_m": 0.45, "prox_ray_floor_m": 0.45,
                "reward_swift_scale": 0.25, "reward_open_water_scale": 0.0, "reward_reverse_action_scale": 0.05,
                "reward_goal_entry_bonus": 50.0, "reward_goal_time_bonus": 50.0, "reward_threading_amplitude": 0.0,
                "reference_reward_progress_scale": 20.0, "reference_reward_gate_bonus": 10.0,
                "reference_reward_terminal_success": 100.0}


def fake_backend(n, **runner_kwargs):
    env_kwargs = {k: runner_kwargs.pop(k) for k in ("nan_at", "code9_at", "nan_obs_at", "detach", "done_period")
                  if k in runner_kwargs}
    mods = T.task_modules(T.namespace_tasks())
    made = {}

    def parse_env_cfg(task_id, device, num_envs):
        info = EP.resolve_journal_task(task_id)
        cfg = T.plan_cfg(mods, info["journal_task"], info["vehicle"], 0)
        cfg.scene = types.SimpleNamespace(num_envs=num_envs)
        cfg.episode_length_s = 60.0                                   # G44: the reward fields of every family
        for name, value in G44_FAKE_CFG.items():
            if info["carrier"] in ("path_following", "station_keeping") and name == "reward_contact_entry_penalty":
                continue
            setattr(cfg, name, value)
        if R.CODE9_PENALTY_FIELD[info["carrier"]] is not None:     # the family cfg field (verify2 C1)
            setattr(cfg, R.CODE9_PENALTY_FIELD[info["carrier"]], 25.0)
        return cfg

    def load_agent_cfg(task_id):
        return T.load_agent_yaml(T.registration_agent_entry(task_id))[0]

    def make_runner(wrapped, cfg):
        made["runner"] = FakeRunner(wrapped, cfg, **runner_kwargs)
        return made["runner"]

    return types.SimpleNamespace(
        package_root=T.namespace_tasks(), torch=torch, parse_env_cfg=parse_env_cfg, load_agent_cfg=load_agent_cfg,
        make_env=lambda task, cfg: FakeEnv(cfg, n, **env_kwargs),
        wrap=lambda env: types.SimpleNamespace(step=env.step, num_envs=n, env=env),
        make_runner=make_runner, made=made)


def plan_for(out, arm="mixture", budget=640, n=4, task="journal:reach"):
    args = args_for(out, "--arm", arm, "--budget-timesteps", str(budget), "--num-envs", str(n),
                    "--checkpoint-every", str(budget // 10), "--device", "cpu", task=task)
    return T.preflight(args, root=T.namespace_tasks(), reference_block=NO_MATCH, code_tree_sha256=tree_sha()), args


@pytest.mark.parametrize("arm,task", [("mixture", "journal:reach"), ("calm", "journal:scatter"),
                                      ("mixture", "journal:path_following")])
def test_completed_run_writes_the_record_last_and_it_validates(scratch, arm, task):
    out = scratch / "run"
    plan, args = plan_for(out, arm=arm, task=task)
    code = T.run_training(plan, args, fake_backend(4))
    assert code == T.EXIT_OK
    record_path = out / "training_record.json"
    record = R.load_record(record_path, run_dir=out)
    assert record["status"] == "completed" and record["completed_timesteps"] == 640
    assert [c["timestep"] for c in record["checkpoints"]] == list(range(64, 641, 64))
    assert record["final_checkpoint"]["source_path"].endswith("agent_640.pt")
    assert record["iterations"] == 640 // plan["rollouts"]
    assert [o["path"] for o in record["agent_config"]["overrides"]] == list(R.ALLOWED_AGENT_OVERRIDES)
    assert record["agent_config"]["config"]["seed"] == SMOKE
    assert record["agent_config"]["config"]["trainer"]["close_environment_at_exit"] is False       # M1
    assert record["preflight_sha256"] == R.file_sha256(out / "preflight.json")                        # m4
    assert record["agent_config"]["yaml_sha256"] == plan["agent"]["yaml_sha256"]
    assert record["agent_config"]["base_config_sha256"] == plan["agent"]["base_config_sha256"]
    assert record["nonfinite_terminal"]["rows_replaced"] == 0
    newest = max((p for p in out.rglob("*") if p.is_file()), key=lambda p: p.stat().st_mtime_ns)
    assert newest == record_path                                           # record written LAST
    beat = json.loads((out / "heartbeat.json").read_text(encoding="ascii"))
    assert beat["status"] == "finishing" and beat["timestep"] == 640
    progress = (out / "progress.log").read_text(encoding="ascii").splitlines()
    assert len(progress) == 100 and progress[-1].startswith("PROGRESS 100%")
    assert json.loads((out / "preflight.json").read_text(encoding="ascii"))["journal_id"] == plan["journal_id"]
    returns = record["training_returns"]
    assert returns["bins"] and returns["first_10pct_mean_return"] is not None
    if arm == "mixture":
        assert record["mixture"]["mixture_sha256"] == plan["mixture"]["mixture_sha256"]
        assert sum(record["mixture"]["realised_episode_starts"].values()) > 4
        assert set(returns["per_component"]) <= set(T.task_modules(T.namespace_tasks()).mixture.COMPONENTS)
        assert len(returns["per_component"]) > 1
    else:
        assert record["mixture"] is None and set(returns["per_component"]) == {"calm"}
    assert record["hazard_admission"]["schema"] == "usvbench.training_admission_log/1"
    with pytest.raises(R.RecordError):                                      # never overwritten
        R.write_record(record, record_path)


def test_divergence_by_non_finite_reward_on_a_non_terminal_row(scratch):
    out = scratch / "run"
    plan, args = plan_for(out)
    code = T.run_training(plan, args, fake_backend(4, nan_at=100))
    assert code == T.EXIT_DIVERGED
    record = R.load_record(out / "training_record.json", run_dir=out)
    assert record["status"] == "diverged" and record["final_checkpoint"] is None
    assert "non-finite reward" in record["divergence"]["reason"] and "not a code-9" in record["divergence"]["reason"]
    assert record["completed_timesteps"] < 640 and record["nonfinite_terminal"]["rows_replaced"] == 0


@pytest.mark.parametrize("task,expected,field", [("journal:reach", -25.0, "reward_contact_entry_penalty"),
                                                 ("journal:path_following", 0.0, None)])
def test_code9_row_reward_is_the_family_failure_penalty_and_the_run_completes(scratch, task, expected, field):
    """Follow-up C1 + verify2 item 4: the NaN reward of a code-9 terminating row is a legal outcome, replaced by the
    family's terminal failure penalty read from the run's env cfg (-reward_contact_entry_penalty, or 0.0 for a
    family whose reward has no termination penalty), stamped with its source."""
    out = scratch / "run"
    plan, args = plan_for(out, task=task)
    assert plan["code9"]["penalty_field"] == field and plan["code9"]["alarm"] == R.code9_alarm_block()
    assert "--code9-reward" not in T.build_parser().format_help()          # no free choice any more
    backend = fake_backend(4, code9_at=101)
    assert T.run_training(plan, args, backend) == T.EXIT_OK
    record = R.load_record(out / "training_record.json", run_dir=out)
    nft = record["nonfinite_terminal"]
    assert record["status"] == "completed" and nft["rows_replaced"] == 1 and nft["replacement_reward"] == expected
    assert nft["replacement_source"]["field"] == field and nft["episodes_finished"] > 300
    assert nft["termination_code"] == 9 and sum(nft["per_component"].values()) == 1
    seen = backend.made["runner"].rewards_seen
    assert all(bool(torch.isfinite(r).all()) for r in seen)                 # the agent never saw the NaN
    fix = json.loads((out / "ppo_fix.json").read_text(encoding="ascii"))          # G44: plus the failure penalty
    penalty = -(fix["failure_terminal_penalty"]["terms"]["A"]
                + fix["failure_terminal_penalty"]["terms"]["B"] * math.hypot(50.0, 44.0))
    assert abs(float(seen[100][0, 0]) - (expected + penalty)) < 1e-3 * max(1.0, abs(penalty))
    assert record["divergence_checks"]["updates_checked"] == 640 // plan["rollouts"]


def test_code9_replacement_reads_the_cfg_and_refuses_a_stale_table():
    cfg = types.SimpleNamespace(reward_contact_entry_penalty=25.0)
    assert T.code9_replacement(cfg, "docking")[0] == -25.0
    cfg.reward_contact_entry_penalty = 40.0
    value, source = T.code9_replacement(cfg, "harbor_mission")
    assert value == -40.0 and source == {"carrier": "harbor_mission", "field": "reward_contact_entry_penalty",
                                         "cfg_value": 40.0, "value_rule": "-env_cfg.reward_contact_entry_penalty"}
    assert T.code9_replacement(types.SimpleNamespace(), "station_keeping")[0] == 0.0
    for cfg, carrier in ((types.SimpleNamespace(), "hazard_nav"),
                         (types.SimpleNamespace(reward_contact_entry_penalty=float("nan")), "path_hazard"),
                         (types.SimpleNamespace(reward_contact_entry_penalty=25.0), "path_following"),
                         (types.SimpleNamespace(), "unknown_family")):
        with pytest.raises(RuntimeError):
            T.code9_replacement(cfg, carrier)
    assert set(R.CODE9_PENALTY_FIELD) == {v["carrier"] for v in
                                          T.task_modules(T.namespace_tasks()).plans.TASK_TABLE.values()}


@pytest.mark.parametrize("code9_steps,status", [((101, 202, 303), "completed"), ((101, 202, 303, 404), "diverged")])
def test_code9_alarm_stops_the_run_as_diverged(scratch, code9_steps, status):
    """verify2 item 4: more than max(3, 1e-4 x finished episodes) code-9 rows stop the run as 'diverged'."""
    out = scratch / "run"
    plan, args = plan_for(out)
    code = T.run_training(plan, args, fake_backend(4, code9_at=code9_steps))
    record = R.load_record(out / "training_record.json", run_dir=out)
    assert record["status"] == status and code == (T.EXIT_OK if status == "completed" else T.EXIT_DIVERGED)
    if status == "diverged":
        assert "code-9 alarm" in record["divergence"]["reason"] and record["completed_timesteps"] == 403
        assert record["nonfinite_terminal"]["rows_replaced"] == 4
    assert R.code9_alarm_limit(0) == 3 and R.code9_alarm_limit(50000) == 5.0


def test_divergence_by_non_finite_observation(scratch):
    out = scratch / "run"
    plan, args = plan_for(out)
    rollouts = plan["rollouts"]
    assert T.run_training(plan, args, fake_backend(4, nan_obs_at=rollouts)) == T.EXIT_DIVERGED
    record = R.load_record(out / "training_record.json", run_dir=out)
    assert "non-finite observation" in record["divergence"]["reason"]


def test_non_finite_observation_on_a_non_done_row_after_a_boundary_is_caught(scratch):
    """verify2 item 3: a NaN observation on a non-done row just after an iteration end (no row done until the next
    one) is caught at the next check from the device-side flag over all rows."""
    out = scratch / "run"
    plan, args = plan_for(out)
    rollouts = plan["rollouts"]
    assert T.run_training(plan, args, fake_backend(4, nan_obs_at=rollouts + 1, done_period=10 ** 9)) == \
        T.EXIT_DIVERGED
    record = R.load_record(out / "training_record.json", run_dir=out)
    assert "non-finite observation" in record["divergence"]["reason"]
    assert record["completed_timesteps"] == 2 * rollouts - 1              # read at the next iteration end


def test_trainer_exception_with_finite_agent_state_and_a_nan_observation_is_diverged(scratch):
    """verify2 item 3: the policy reading a NaN observation raises in act() before any check ran; the agent state is
    finite, the last observation is not: 'diverged', not an infrastructure failure."""
    out = scratch / "run"
    plan, args = plan_for(out)
    t = next(t for t in range(plan["rollouts"] + 1, 640) if t % 7 in (1, 2, 3) and t % plan["rollouts"])
    assert T.run_training(plan, args, fake_backend(4, nan_obs_at=t, raise_on_nan_obs=True)) == T.EXIT_DIVERGED
    record = R.load_record(out / "training_record.json", run_dir=out)
    assert "non-finite last observation after ValueError" in record["divergence"]["reason"], record["divergence"]
    assert not (out / "STATUS.json").exists()


def test_non_finite_action_is_diverged_before_env_step(scratch):
    """verify2 item 5: a NaN action never reaches env.step."""
    out = scratch / "run"
    plan, args = plan_for(out)
    backend = fake_backend(4, nan_action_at=77)
    assert T.run_training(plan, args, backend) == T.EXIT_DIVERGED
    record = R.load_record(out / "training_record.json", run_dir=out)
    assert "non-finite action at timestep 77" in record["divergence"]["reason"]
    assert record["completed_timesteps"] == 77 and len(backend.made["runner"].rewards_seen) == 77


@pytest.mark.parametrize("kwargs,reason", [
    ({"nan_loss_at": 128}, "non-finite Loss / Policy loss"),            # track_data hook, every update
    ({"nan_params_at": 192}, "non-finite agent state (policy)"),        # parameters after every update
    ({"nan_params_at": 192, "raise_after_nan_params": True}, "after ValueError"),   # NaN policy -> torch ValueError
])
def test_divergence_by_losses_parameters_and_nan_policy_exceptions(scratch, kwargs, reason):
    """Follow-up M2: diverged (exit 4, record), never an infrastructure failure."""
    out = scratch / "run"
    plan, args = plan_for(out)
    assert T.run_training(plan, args, fake_backend(4, **kwargs)) == T.EXIT_DIVERGED
    record = R.load_record(out / "training_record.json", run_dir=out)
    assert record["status"] == "diverged" and reason in record["divergence"]["reason"], record["divergence"]
    assert not (out / "STATUS.json").exists()


def test_env_must_carry_the_mixture_and_the_adapters(scratch):
    """Follow-up m5: an env that dropped the run's mixture fails loudly (infrastructure failure, no record)."""
    out = scratch / "run"
    plan, args = plan_for(out)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        assert T.run_training(plan, args, fake_backend(4, detach=True)) == T.EXIT_INFRA
    assert "training mixture" in json.loads((out / "STATUS.json").read_text(encoding="ascii"))["error"]
    assert not (out / "training_record.json").exists()


def test_divergence_by_non_finite_final_parameters(scratch):
    out = scratch / "run"
    plan, args = plan_for(out)
    assert T.run_training(plan, args, fake_backend(4, bad_final=True)) == T.EXIT_DIVERGED
    record = R.load_record(out / "training_record.json", run_dir=out)
    assert record["divergence"]["reason"].startswith("non-finite parameters")
    assert not (out / "final.pt").exists()


@pytest.mark.parametrize("kwargs,message", [({"fail_at": 300}, "simulated host failure"),
                                            ({"skip_final": True}, "final checkpoint")])
def test_infrastructure_failures_leave_no_training_record(scratch, kwargs, message):
    out = scratch / "run"
    plan, args = plan_for(out)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        code = T.run_training(plan, args, fake_backend(4, **kwargs))
    assert code == T.EXIT_INFRA
    assert not (out / "training_record.json").exists()
    status = json.loads((out / "STATUS.json").read_text(encoding="ascii"))
    assert status["status"] == "infrastructure_failure" and message in status["error"]
    assert (out / "preflight.json").is_file()


def test_main_always_ends_through_finish(scratch, monkeypatch):
    """verify2 item 2: an exception before the run (missing registry, failing preflight, failing backend after the
    app started) never escapes main(); finish() gets the right code and the launched app."""
    calls = []
    monkeypatch.setattr(T, "finish", lambda code, backend: calls.append((code, backend)) or code)
    argv = ["--task", "journal:reach", "--vessel", "blueboat", "--training-seed", str(SMOKE)]
    code, _, err = run_main(argv + ["--out", str(scratch / "a"), "--seed-registry", str(scratch / "missing.json")],
                            backend=fake_backend(2))
    assert code == T.EXIT_REFUSED and "is missing" in err and calls[-1][0] == T.EXIT_REFUSED
    real_preflight = T.preflight

    def broken_preflight(*args, **kwargs):
        raise FileNotFoundError("simulated missing file")
    monkeypatch.setattr(T, "preflight", broken_preflight)
    code, _, err = run_main(argv + ["--out", str(scratch / "b")], backend=fake_backend(2))
    assert code == T.EXIT_INFRA and "INFRASTRUCTURE FAILURE before the run" in err and calls[-1][0] == T.EXIT_INFRA
    monkeypatch.setattr(T, "preflight", real_preflight)
    app = object()

    def failing_backend(app_argv, device):
        T._LAUNCHED["app"] = app                      # AppLauncher started, then an import failed
        raise ImportError("simulated isaaclab_tasks import failure")
    monkeypatch.setattr(T, "isaac_backend", failing_backend)
    monkeypatch.setattr(T, "_LAUNCHED", {})
    code, _, err = run_main(argv + ["--out", str(scratch / "c")])
    assert code == T.EXIT_INFRA and calls[-1][1].app is app
    monkeypatch.setattr(T, "run_training", lambda plan, args, backend: (_ for _ in ()).throw(OSError("disk")))
    code, _, err = run_main(argv + ["--out", str(scratch / "d")], backend=fake_backend(2))
    assert code == T.EXIT_INFRA and calls[-1][0] == T.EXIT_INFRA and "disk" in err


def test_smoke_seed_with_another_registry_is_refused(scratch):
    """verify2 item 1: training_smoke runs use the in-tree registry too (a registry relabelling an evaluation seed
    as smoke no longer passes preflight)."""
    registry = SR.load_registry()
    registry["namespaces"]["training_smoke"]["blocks"][0]["seed"] = 41041            # the REV probe's relabel
    forged = scratch / "smoke_registry.json"
    forged.write_text(json.dumps(registry, indent=2), encoding="ascii")
    code, _, err = run_main(["--task", "journal:reach", "--vessel", "blueboat", "--training-seed", "41041",
                             "--purpose", "training_smoke", "--out", str(scratch / "run"), "--dry-run",
                             "--seed-registry", str(forged)])
    assert code == T.EXIT_REFUSED, err
    cosmetic = SR.load_registry()                                          # any other digest, valid registry
    cosmetic["status_rule"] = cosmetic["status_rule"] + " (edited)"
    forged.write_text(json.dumps(cosmetic, indent=2), encoding="ascii")
    code, _, err = run_main(["--task", "journal:reach", "--vessel", "blueboat", "--training-seed", str(SMOKE),
                             "--out", str(scratch / "run"), "--dry-run", "--seed-registry", str(forged)])
    assert code == T.EXIT_REFUSED and "in-tree seed registry" in err and "training_smoke" in err, err
    same = scratch / "same_registry.json"                                  # the in-tree content passes
    same.write_text(json.dumps(SR.load_registry(), indent=2), encoding="ascii")
    code, _, err = run_main(["--task", "journal:reach", "--vessel", "blueboat", "--training-seed", str(SMOKE),
                             "--out", str(scratch / "run"), "--dry-run", "--seed-registry", str(same)])
    assert code == T.EXIT_OK, err


def test_main_run_path_with_backend_refuses_existing_out(scratch):
    (scratch / "run").mkdir()
    code, _, err = run_main(["--task", "journal:reach", "--vessel", "blueboat", "--training-seed", str(SMOKE),
                             "--out", str(scratch / "run")], backend=fake_backend(2))
    assert code == T.EXIT_REFUSED and "exists" in err


# ------------------------------------------------------------------ record validate()

def completed_record(scratch):
    out = scratch / "run"
    plan, args = plan_for(out)
    assert T.run_training(plan, args, fake_backend(4)) == T.EXIT_OK
    return out, json.loads((out / "training_record.json").read_text(encoding="ascii"))


def test_record_validates_the_code9_source_and_alarm(scratch):
    """verify2 item 4: the record's code-9 replacement must be the family's declared penalty, the alarm the declared
    one, and a completed run within the alarm limit."""
    out, record = completed_record(scratch)
    nft = record["nonfinite_terminal"]
    assert nft["replacement_reward"] == -25.0 and nft["replacement_source"]["field"] == "reward_contact_entry_penalty"
    for path, value in ((("replacement_reward",), -2.5), (("replacement_source", "field"), None),
                        (("alarm", "min_rows"), 10), (("rows_replaced",), 4), (("episodes_finished",), -1)):
        broken = copy.deepcopy(record)
        target = broken["nonfinite_terminal"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        if path == ("rows_replaced",):
            target["per_component"] = {"calm": 4}
        with pytest.raises(R.RecordError):
            R.validate(broken)


def test_record_round_trip_fields_and_hash_recomputation(scratch):
    out, record = completed_record(scratch)
    assert set(record) == set(R.RECORD_FIELDS)
    schema = R.schema_document()
    assert schema["required"] == list(R.RECORD_FIELDS) and set(schema["properties"]) == set(R.RECORD_FIELDS)
    R.validate(json.loads(json.dumps(record)), run_dir=out)
    for field in R.RECORD_FIELDS:
        broken = copy.deepcopy(record)
        del broken[field]
        with pytest.raises(R.RecordError):
            R.validate(broken)
    tampered = copy.deepcopy(record)
    tampered["checkpoints"][0]["sha256"] = "0" * 64
    with pytest.raises(R.RecordError, match="hash mismatch"):
        R.validate(tampered, run_dir=out)
    (out / "preflight.json").write_bytes(b"{}")
    with pytest.raises(R.RecordError, match="preflight.json hash"):
        R.validate(record, run_dir=out)
    (out / "final.pt").write_bytes(b"swapped")
    with pytest.raises(R.RecordError, match="hash"):
        R.validate(record, run_dir=out)


def test_record_registry_stamp_is_checked_against_the_registry(scratch):
    """Follow-up M4: validate() re-resolves the seed in the registry (in-tree unless given)."""
    _, record = completed_record(scratch)
    R.validate(record)
    for name, value in (("registry_sha256", "0" * 64), ("namespace", "training"), ("block_id", "smoke-90002"),
                        ("registry_status", "proposal")):          # G52: the in-tree registry is frozen
        broken = copy.deepcopy(record)
        broken["registry"][name] = value
        with pytest.raises(R.RecordError, match="registry"):
            R.validate(broken)
    with pytest.raises(R.RecordError, match="registry"):
        R.validate(record, registry=frozen_registry())                     # another registry digest
    campaign = copy.deepcopy(record)
    campaign["purpose"], campaign["training_seed"] = "training", 42
    with pytest.raises(R.RecordError):
        R.validate(campaign)
    broken = copy.deepcopy(record)
    broken["release"] = {"release_id": "R", "record_sha256": "a" * 64, "code_tree_sha256": "b" * 64}
    with pytest.raises(R.RecordError, match="code tree"):
        R.validate(broken)
    for mutate in (lambda r: r["nonfinite_terminal"].__setitem__("replacement_reward", float("nan")),
                   lambda r: r["nonfinite_terminal"].__setitem__("rows_replaced", 3),
                   lambda r: r["agent_config"].__setitem__("yaml_sha256", "c" * 64)):
        broken = copy.deepcopy(record)
        mutate(broken)
        with pytest.raises(R.RecordError):
            R.validate(broken)


def test_record_refuses_a_mixture_digest_mismatch_and_foreign_overrides(scratch):
    _, record = completed_record(scratch)
    for mutate in (lambda r: r["mixture"].__setitem__("mixture_sha256", "1" * 64),
                   lambda r: r["mixture"].__setitem__("calm_slots", 2),
                   lambda r: r["mixture"]["presets"]["waves"]["values"]["waves"].__setitem__("hs_m", 0.25),
                   lambda r: r["agent_config"]["config"]["agent"].__setitem__("learning_rate", 1.0),
                   lambda r: r["agent_config"]["overrides"].append({"path": "agent.rollouts", "value": 16}),
                   lambda r: r.__setitem__("completed_timesteps", 320),
                   lambda r: r.__setitem__("status", "unknown")):
        broken = copy.deepcopy(record)
        mutate(broken)
        with pytest.raises(R.RecordError):
            R.validate(broken)


# ------------------------------------------------------------------ P4 evaluator refusals

def test_evaluators_refuse_training_only_cfg_content():
    TM = T.task_modules(T.namespace_tasks()).mixture
    mixture = TM.TrainingMixture(seed=SMOKE, task="reach", vessel="blueboat_v2", appearing_wired=False,
                                 purpose="training_smoke")
    for assignment in ("journal_training_mixture=1", "journal_appearing_pool=x", "journal_d7_pool=x"):
        with pytest.raises(EP.OverrideError, match="training-only"):
            EP._check_generic_assignments([assignment], {})
    args = types.SimpleNamespace(eval_seed=42, level=0, num_envs=2, device=None, extra_sets=[])
    with pytest.raises(EP.OverrideError, match="journal_training_mixture"):
        EP.resolve_protocol(types.SimpleNamespace(journal_training_mixture=mixture), args)
    adapter = types.SimpleNamespace(TRAINING_ADAPTER=True)
    for field in ("journal_d7_pool", "journal_appearing_pool"):
        with pytest.raises(EP.OverrideError, match="online-admission"):
            EP.resolve_protocol(types.SimpleNamespace(**{field: adapter}), args)
    EP.refuse_training_cfg(types.SimpleNamespace(journal_d7_pool=object(), journal_training_mixture=None))


def test_preflight_protocol_refuses_training_fields_for_journal_and_legacy_ids():
    from unittest.mock import patch
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--set", dest="extra_sets", action="append", default=[])
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=128)
    EP.add_protocol_arguments(parser)
    with patch.dict("os.environ", {}, clear=True):
        for task in ("Isaac-USV-Reach-BlueBoat-Journal-Direct-v1", "Isaac-USV-HazardNav-Direct-v1"):
            for field in ("journal_training_mixture", "journal_appearing_pool"):
                args = parser.parse_args(["--task", task, "--set", field + "=1"])
                with pytest.raises(EP.OverrideError, match="training-only"):
                    EP.preflight_protocol(args, ["x", "--set", field + "=1"])


def test_every_certificate_writer_runs_resolve_protocol_on_its_env_cfg_before_gym_make():
    """P4 (strengthened, follow-up): in every certificate writer, each gym.make(..., cfg=X) is preceded, in the same
    function (or module body), by resolve_protocol(X, ...) on the SAME cfg variable, so refuse_training_cfg runs on
    the cfg the env is built from, before the env exists."""
    import ast
    writers = ("eval_v6_frozen.py", "eval_mppi.py", "classical_planner_pid.py", "classical_baseline.py",
               "classical_harbour.py", "eval_imbalance.py", "eval_obs_bridge.py", "eval_composite.py",
               "eval_hazard_nav.py", "eval_cross_champion.py", "convergence_curve.py")

    def scopes(tree):
        yield tree
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield node

    def direct_calls(scope):
        out, stack = [], list(ast.iter_child_nodes(scope))
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, ast.Call):
                out.append(node)
            stack.extend(ast.iter_child_nodes(node))
        return out

    checked = 0
    for name in writers:
        tree = ast.parse((SCRIPTS / name).read_text(encoding="utf-8"))
        for scope in scopes(tree):
            calls = direct_calls(scope)
            resolves = [(c.lineno, c.args[0].id) for c in calls
                        if getattr(c.func, "id", getattr(c.func, "attr", None)) == "resolve_protocol"
                        and c.args and isinstance(c.args[0], ast.Name)]
            for call in calls:
                if getattr(call.func, "attr", None) == "make" and getattr(call.func.value, "id", None) == "gym":
                    cfg = [k.value.id for k in call.keywords if k.arg == "cfg" and isinstance(k.value, ast.Name)]
                    assert cfg, (name, call.lineno)
                    assert any(line < call.lineno and var == cfg[0] for line, var in resolves), (name, call.lineno)
                    checked += 1
    assert checked >= len(writers)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
