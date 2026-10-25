"""PPO training record (G19b, IMPLEMENTATION_PLAN P6): schema usvbench.ppo_training_record/1, builder, validate().

One JSON file per training run (one task, one vessel, one training seed), written by
scripts/train_journal_ppo.py as the LAST file of the run directory. It is the evidence of the run that the gate
ledger v2 (seed_registry.py, ``training_record_sha256``) and the later policy registry (P8) cite. Fields
(RECORD_FIELDS, JSON schema ppo_training_record.schema.json beside this file):

  schema, task, journal_id, vessel, vehicle, carrier_family, arm ("mixture" | "calm"),
  training_seed, candidate_index, purpose, registry (seed_registry.resolve_training_block stamp),
  release ({release_id, record_path, record_sha256} or null), code_tree_sha256,
  reference_environment (reference_environment.reference_block()), re1_match,
  host ({gpu, driver, cpu, os, python, hostname_sha256}), host_class, concurrency,
  preflight_sha256 (of preflight.json, the run's first file),
  agent_config ({entry_point, yaml_path, yaml_sha256, base_config_sha256, config, sha256, overrides [{path, value}]}),
  budget_timesteps, completed_timesteps, num_envs, rollouts, iterations,
  mixture (TrainingMixture.stamp(realised counts) or null for the calm arm),
  hazard_admission (TrainingHazardCache.admission_log() or null),
  checkpoints [{timestep, path, sha256}], final_checkpoint ({path, sha256, source_path, source_sha256} or null),
  status ("completed" | "diverged"), divergence (null or {reason, timestep}),
  nonfinite_terminal ({rule, termination_code 9, replacement_reward, replacement_source {carrier, field,
                      cfg_value, value_rule}, rows_replaced, episodes_finished, alarm, per_component}: code-9 rows;
                      replacement = the family's terminal failure penalty (CODE9_PENALTY_FIELD); a completed run
                      stays within code9_alarm_limit),
  divergence_checks ({updates_checked, loss_values_checked, observation_check, agent_state}),
  wall_time_s, started_utc, finished_utc,
  training_returns ({bins: [{percent, episodes, mean_return}], per_component: {name: {episodes, mean_return,
                     termination_codes}}, first_10pct_mean_return, last_10pct_mean_return}),
  log_digests ({relative path: sha256} of the skrl event files, progress log and heartbeat).

Paths are relative to the run directory. ``validate(record, run_dir=None)`` checks every field, recomputes the
mixture digest from its own spec (TrainingMixture rebuilt from the stamp), the agent-config digest, that every
override value is the value in the config, the budget / status rules, the seed-registry stamp against the registry
(follow-up M4: in-tree scripts/seed_registry.json unless one is passed; campaign records need it frozen, and a
release that pins a code tree must pin this one) and, with ``run_dir``, every file hash (preflight.json too).
Pure Python; imports no simulator package (the mixture module is loaded by path from tasks/_shared).
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types

SCHEMA = "usvbench.ppo_training_record/1"
SCHEMA_FILE = Path(__file__).resolve().with_name("ppo_training_record.schema.json")
STATUSES = ("completed", "diverged")
ARMS = ("mixture", "calm")
PURPOSES = ("training", "training_smoke")
VESSELS = ("blueboat", "wamv")
RECORD_FIELDS = (
    "schema", "task", "journal_id", "vessel", "vehicle", "carrier_family", "arm",
    "training_seed", "candidate_index", "purpose", "registry", "release", "code_tree_sha256",
    "reference_environment", "re1_match", "host", "host_class", "concurrency", "preflight_sha256", "agent_config",
    "budget_timesteps", "completed_timesteps", "num_envs", "rollouts", "iterations",
    "mixture", "hazard_admission", "checkpoints", "final_checkpoint", "status", "divergence",
    "nonfinite_terminal", "divergence_checks",
    "wall_time_s", "started_utc", "finished_utc", "training_returns", "log_digests")
# C1 controller decision (REV-G19b verify2 item 4): a code-9 row's non-finite reward is replaced by the family's
# terminal failure penalty = the reward the family assigns to a collision termination, read from the run's env cfg
# (hazard_nav_env_cfg.py:266, path_hazard_env_cfg.py:265, docking_env_cfg.py:362, harbor_mission_env_cfg.py:209:
# reward_contact_entry_penalty = 25.0, applied as -25 on the contact-entry step that ends the episode). The
# path_following and station_keeping journal rewards (journal_path_following.py _journal_get_rewards,
# station_keeping_env.py _journal_get_rewards) carry no termination penalty: their collision / out-of-bounds /
# failure exits are paid by the step's shaping alone, so a code-9 row gets 0.0 there, the same as those exits.
CODE9_PENALTY_FIELD = {"hazard_nav": "reward_contact_entry_penalty", "path_hazard": "reward_contact_entry_penalty",
                       "docking": "reward_contact_entry_penalty", "harbor_mission": "reward_contact_entry_penalty",
                       "path_following": None, "station_keeping": None}
CODE9_NO_PENALTY_RULE = ("0.0: the family's journal reward has no termination penalty (its collision / "
                         "out-of-bounds / failure exits are paid by the step shaping alone)")
CODE9_REPLACEMENT_RULE = ("the family's terminal failure penalty: -env_cfg.<CODE9_PENALTY_FIELD[carrier]> (the "
                          "reward the family assigns to a collision termination), or 0.0 for a family without one")
# Code-9 alarm: up to 3 code-9 rows per run are tolerated whatever the episode count; beyond that the rate must stay
# at or below 1 per 10,000 finished episodes, else the run stops as 'diverged'.
CODE9_ALARM_MIN_ROWS = 3
CODE9_ALARM_RATE = 1.0e-4
CODE9_ALARM_RULE = "max(3, 1e-4 x finished episodes)"


def code9_alarm_limit(episodes_finished):
    """Largest code-9 row count a run may reach after ``episodes_finished`` episodes."""
    return max(float(CODE9_ALARM_MIN_ROWS), CODE9_ALARM_RATE * float(episodes_finished))


def code9_alarm_block():
    return {"min_rows": CODE9_ALARM_MIN_ROWS, "rate_per_episode": CODE9_ALARM_RATE, "rule": CODE9_ALARM_RULE}


ALLOWED_AGENT_OVERRIDES = ("seed", "trainer.timesteps", "trainer.close_environment_at_exit", "agent.experiment.directory",
                           "agent.experiment.experiment_name", "agent.experiment.checkpoint_interval",
                           "agent.experiment.write_interval", "agent.experiment.wandb")


class RecordError(ValueError):
    """A training record is malformed or does not match its evidence."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("ascii")).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def get_path(config, path):
    node = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(path)
        node = node[part]
    return node


def set_path(config, path, value):
    parts = path.split(".")
    node = config
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise RecordError("agent config has no section %r for override %s" % (part, path))
        node = node[part]
    node[parts[-1]] = value


# ------------------------------------------------------------------ mixture (by path)
_MIXTURE = None


def mixture_module():
    """tasks/_shared/conditions/training_mixture.py under a private package (no family __init__ runs)."""
    global _MIXTURE
    if _MIXTURE is None:
        shared = Path(__file__).resolve().parents[1] / "tasks" / "_shared"
        name = "_g19b_record_shared"
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(shared)]
            sys.modules[name] = package
        _MIXTURE = importlib.import_module(name + ".conditions.training_mixture")
    return _MIXTURE


_REGISTRY = None


def seed_registry_module():
    """scripts/seed_registry.py of this tree (loaded by path)."""
    global _REGISTRY
    if _REGISTRY is None:
        spec = importlib.util.spec_from_file_location("_g19b_record_seed_registry",
                                                      Path(__file__).resolve().with_name("seed_registry.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _REGISTRY = module
    return _REGISTRY


def check_registry_stamp(record, registry=None):
    """Follow-up M4: the record's registry stamp must be what the registry resolves for its seed and purpose."""
    module = seed_registry_module()
    reg = module.load_registry() if registry is None else registry
    try:
        resolved = module.resolve_training_block(record["training_seed"], record["purpose"], reg)
    except module.RegistryError as exc:
        raise RecordError("registry check: %s" % exc) from None
    for name in ("registry_sha256", "namespace", "block_id", "seed", "purpose", "registry_status", "gate_status"):
        if record["registry"].get(name) != resolved.get(name):
            raise RecordError("registry stamp %s %r differs from the registry (%r)"
                              % (name, record["registry"].get(name), resolved.get(name)))
    return resolved


def recompute_mixture_digest(stamp):
    """Digest of the TrainingMixture rebuilt from the stamp's own fields; also checks the spec equality."""
    tm = mixture_module()
    m = tm.TrainingMixture(seed=int(stamp["seed"]), task=stamp["task"], vessel=stamp["vessel"],
                           appearing_wired=bool(stamp["appearing_wired"]), calm_slots=int(stamp["calm_slots"]),
                           purpose=stamp["purpose"], protocol_version=int(stamp["scenario_protocol_version"]))
    spec = m.spec()
    for key, value in spec.items():
        if stamp.get(key) != json.loads(canonical(value)):
            raise RecordError("mixture stamp field %r differs from the rebuilt mixture" % key)
    return m.digest()


# ------------------------------------------------------------------ returns
class ReturnTracker:
    """Per-episode training returns: per 1 % of the budget and per mixture component (smoke L1, reward hacking)."""

    def __init__(self, budget_timesteps, num_envs):
        self.budget = int(budget_timesteps)
        self.num_envs = int(num_envs)
        self.running = [0.0] * self.num_envs
        self.bins = {}
        self.per_component = {}

    def step(self, timestep, rewards, dones, components, codes=None):
        """``rewards`` / ``dones`` per env (lists); ``components`` = the component of the FINISHED episode per
        env (only read where done); ``codes`` = termination code per env or None."""
        percent = min(99, int(100 * int(timestep) // max(1, self.budget)))
        for i in range(self.num_envs):
            self.running[i] += float(rewards[i])
            if not dones[i]:
                continue
            value = self.running[i]
            self.running[i] = 0.0
            if not math.isfinite(value):
                continue
            entry = self.bins.setdefault(percent, [0, 0.0])
            entry[0] += 1
            entry[1] += value
            name = components[i] if components is not None else "calm"
            comp = self.per_component.setdefault(name, {"episodes": 0, "return_sum": 0.0, "termination_codes": {}})
            comp["episodes"] += 1
            comp["return_sum"] += value
            if codes is not None:
                code = str(int(codes[i]))
                comp["termination_codes"][code] = comp["termination_codes"].get(code, 0) + 1

    def summary(self):
        bins = [{"percent": p, "episodes": n, "mean_return": s / n} for p, (n, s) in sorted(self.bins.items())]

        def window(lo, hi):
            n = sum(b["episodes"] for b in bins if lo <= b["percent"] < hi)
            s = sum(b["mean_return"] * b["episodes"] for b in bins if lo <= b["percent"] < hi)
            return s / n if n else None
        per = {name: {"episodes": c["episodes"], "mean_return": c["return_sum"] / c["episodes"],
                      "termination_codes": dict(sorted(c["termination_codes"].items()))}
               for name, c in sorted(self.per_component.items()) if c["episodes"]}
        return {"bins": bins, "per_component": per, "first_10pct_mean_return": window(0, 10),
                "last_10pct_mean_return": window(90, 100)}


# ------------------------------------------------------------------ validate
def _require(cond, message):
    if not cond:
        raise RecordError(message)


def validate(record, run_dir=None, registry=None):
    """Fail closed unless ``record`` is a complete, self-consistent training record; returns it."""
    _require(isinstance(record, dict), "a training record is a JSON object")
    _require(set(record) == set(RECORD_FIELDS), "record fields differ from RECORD_FIELDS: missing %s, extra %s"
             % (sorted(set(RECORD_FIELDS) - set(record)), sorted(set(record) - set(RECORD_FIELDS))))
    _require(record["schema"] == SCHEMA, "unsupported record schema")
    _require(record["arm"] in ARMS, "arm must be mixture or calm")
    _require(record["purpose"] in PURPOSES, "purpose must be training or training_smoke")
    _require(record["vessel"] in VESSELS, "vessel must be blueboat or wamv")
    _require(record["status"] in STATUSES, "status must be completed or diverged")
    for name in ("training_seed", "budget_timesteps", "completed_timesteps", "num_envs", "rollouts",
                 "iterations", "concurrency"):
        _require(type(record[name]) is int and record[name] >= 0, name + " must be a nonnegative integer")
    _require(record["budget_timesteps"] > 0 and record["num_envs"] > 0 and record["rollouts"] > 0,
             "budget, num_envs and rollouts must be positive")
    _require(record["iterations"] == record["completed_timesteps"] // record["rollouts"],
             "iterations must equal completed_timesteps // rollouts")
    stamp = record["registry"]
    _require(isinstance(stamp, dict) and stamp.get("seed") == record["training_seed"]
             and stamp.get("purpose") == record["purpose"] and _is_sha(stamp.get("registry_sha256")),
             "registry stamp must name the training seed, its purpose and the registry SHA-256")
    if record["purpose"] == "training":
        _require(record["candidate_index"] == stamp.get("candidate_index"), "candidate_index differs")
        _require(stamp.get("registry_status") == "frozen", "campaign records need a frozen registry")
        _require(record["re1_match"] is True, "campaign training must run in RE-1 (CONTEXT_R4 D28)")
        _require(isinstance(record["release"], dict), "campaign records name their release")
    else:
        _require(record["candidate_index"] is None, "smoke records have no candidate index")
    if record["release"] is not None:
        _require(isinstance(record["release"].get("release_id"), str) and _is_sha(record["release"].get(
            "record_sha256")), "release needs release_id and record_sha256")
        _require(record["release"].get("code_tree_sha256") in (None, record["code_tree_sha256"]),
                 "the release record pins another code tree (follow-up M4)")
    check_registry_stamp(record, registry)
    _require(_is_sha(record["preflight_sha256"]), "preflight_sha256 required")
    nft = record["nonfinite_terminal"]
    _require(isinstance(nft, dict) and nft.get("termination_code") == 9
             and isinstance(nft.get("replacement_reward"), (int, float))
             and math.isfinite(float(nft["replacement_reward"])) and type(nft.get("rows_replaced")) is int
             and nft["rows_replaced"] >= 0 and sum(nft.get("per_component", {}).values()) == nft["rows_replaced"],
             "nonfinite_terminal must state code 9, a finite replacement reward and its counts")
    source = nft.get("replacement_source")
    carrier = record.get("carrier_family")
    _require(carrier in CODE9_PENALTY_FIELD and isinstance(source, dict) and source.get("carrier") == carrier
             and source.get("field") == CODE9_PENALTY_FIELD[carrier],
             "nonfinite_terminal.replacement_source must name the family's declared penalty field")
    if source["field"] is None:
        _require(float(nft["replacement_reward"]) == 0.0, "a family without a termination penalty replaces by 0.0")
    else:
        cfg_value = source.get("cfg_value")
        _require(isinstance(cfg_value, (int, float)) and cfg_value > 0
                 and float(nft["replacement_reward"]) == -float(cfg_value),
                 "the code-9 replacement must be -env_cfg.%s" % source["field"])
    _require(nft.get("alarm") == code9_alarm_block(), "nonfinite_terminal.alarm must be the declared alarm")
    _require(type(nft.get("episodes_finished")) is int and nft["episodes_finished"] >= 0,
             "nonfinite_terminal.episodes_finished required")
    if record["status"] == "completed":
        _require(nft["rows_replaced"] <= code9_alarm_limit(nft["episodes_finished"]),
                 "a completed run stays within the code-9 alarm limit %s" % CODE9_ALARM_RULE)
    checks = record["divergence_checks"]
    _require(isinstance(checks, dict) and type(checks.get("updates_checked")) is int,
             "divergence_checks.updates_checked required")
    _require(_is_sha(record["code_tree_sha256"]), "code_tree_sha256 required")
    ref = record["reference_environment"]
    _require(isinstance(ref, dict) and ref.get("match") is record["re1_match"], "re1_match must equal the block")
    agent = record["agent_config"]
    _require(isinstance(agent, dict) and isinstance(agent.get("config"), dict), "agent_config.config required")
    _require(_is_sha(agent.get("yaml_sha256")) and _is_sha(agent.get("base_config_sha256"))
             and isinstance(agent.get("yaml_path"), str), "agent_config needs yaml_path, yaml_sha256, base_config_sha256")
    yaml_file = Path(__file__).resolve().parents[1] / agent["yaml_path"]
    if yaml_file.is_file():
        _require(file_sha256(yaml_file) == agent["yaml_sha256"], "agent yaml hash differs from this tree's file")
    _require(agent.get("sha256") == digest(agent["config"]), "agent config digest mismatch")
    overrides = agent.get("overrides")
    _require(isinstance(overrides, list) and [o["path"] for o in overrides] == list(ALLOWED_AGENT_OVERRIDES),
             "the agent overrides are exactly %s, in order" % (ALLOWED_AGENT_OVERRIDES,))
    for item in overrides:
        try:
            value = get_path(agent["config"], item["path"])
        except KeyError:
            raise RecordError("override %s is not in the agent config" % item["path"]) from None
        _require(value == item["value"], "override %s differs from the config value" % item["path"])
    _require(agent["config"].get("seed") == record["training_seed"], "agent seed must be the training seed")
    _require(get_path(agent["config"], "trainer.timesteps") == record["budget_timesteps"],
             "trainer.timesteps must be the budget")
    mixture = record["mixture"]
    if record["arm"] == "mixture":
        _require(isinstance(mixture, dict) and _is_sha(mixture.get("mixture_sha256")), "mixture stamp required")
        _require(recompute_mixture_digest(mixture) == mixture["mixture_sha256"], "mixture digest mismatch")
        _require(mixture["seed"] == record["training_seed"] and mixture["purpose"] == record["purpose"],
                 "mixture seed / purpose differ from the record")
        starts = mixture.get("realised_episode_starts")
        _require(isinstance(starts, dict) and all(type(v) is int and v >= 0 for v in starts.values()),
                 "mixture realised_episode_starts required")
    else:
        _require(mixture is None, "the calm arm carries no mixture")
    checkpoints = record["checkpoints"]
    _require(isinstance(checkpoints, list), "checkpoints must be a list")
    steps = [c.get("timestep") for c in checkpoints]
    _require(steps == sorted(steps) and len(set(steps)) == len(steps), "checkpoints ordered by timestep")
    for c in checkpoints:
        _require(type(c.get("timestep")) is int and isinstance(c.get("path"), str) and _is_sha(c.get("sha256")),
                 "checkpoint entries need timestep, path and sha256")
    final = record["final_checkpoint"]
    if record["status"] == "completed":
        _require(record["completed_timesteps"] == record["budget_timesteps"], "a completed run reaches its budget")
        _require(isinstance(final, dict) and _is_sha(final.get("sha256")), "a completed run has a final checkpoint")
        _require(final["sha256"] == final.get("source_sha256"), "final.pt must be a copy of the last checkpoint")
        _require(bool(checkpoints) and checkpoints[-1]["timestep"] == record["budget_timesteps"]
                 and checkpoints[-1]["sha256"] == final["sha256"],
                 "the final checkpoint is the checkpoint at the budget (last-checkpoint rule)")
        _require(record["divergence"] is None, "a completed run has no divergence")
    else:
        _require(isinstance(record["divergence"], dict) and isinstance(record["divergence"].get("reason"), str),
                 "a diverged run states its divergence")
    returns = record["training_returns"]
    _require(isinstance(returns, dict) and isinstance(returns.get("bins"), list), "training_returns required")
    _require(isinstance(record["log_digests"], dict) and all(_is_sha(v) for v in record["log_digests"].values()),
             "log_digests are SHA-256 values")
    if run_dir is not None:
        root = Path(run_dir)
        _require(file_sha256(root / "preflight.json") == record["preflight_sha256"], "preflight.json hash mismatch")
        for c in checkpoints:
            _require(file_sha256(root / c["path"]) == c["sha256"], "checkpoint %s hash mismatch" % c["path"])
        if final is not None:
            _require(file_sha256(root / final["path"]) == final["sha256"], "final checkpoint hash mismatch")
        for rel, sha in record["log_digests"].items():
            _require(file_sha256(root / rel) == sha, "log %s hash mismatch" % rel)
    return record


def write_record(record, path):
    """Validate and write (canonical, indented, ASCII); returns the file SHA-256. Refuses to overwrite."""
    validate(record)
    path = Path(path)
    if path.exists():
        raise RecordError("refusing to overwrite %s" % path)
    text = json.dumps(record, indent=1, sort_keys=True, allow_nan=False) + "\n"
    path.write_bytes(text.encode("ascii"))
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def load_record(path, run_dir=None, registry=None):
    record = json.loads(Path(path).read_text(encoding="ascii"))
    return validate(record, run_dir, registry)


def schema_document():
    return json.loads(SCHEMA_FILE.read_text(encoding="ascii"))


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="validate a PPO training record")
    parser.add_argument("record", type=Path)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        record = load_record(args.record, args.run_dir)
    except (ValueError, OSError, KeyError) as exc:
        print("ppo_training_record: invalid: %s" % exc, file=sys.stderr)
        return 2
    print(json.dumps({"valid": True, "status": record["status"], "sha256": file_sha256(args.record)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
