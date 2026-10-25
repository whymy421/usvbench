"""Training-only condition mixture for journal PPO training (G19 prototype, rebased on T-base v2.5 by G19b).

Owner decision T1.33 option 2 (2026-10-02): PPO training mixes MILD doses of
every suite into training, the obstacle suites included, so that every
disturbance type has been seen at a mild level; evaluation D2-D4 measures
robustness beyond the training range. This module defines exactly which
levels are in training and assigns ONE component to every training episode:

    calm                       (no suite active)
    waves | wind | current     (the suite at its D1 preset)
    actuator.<cap|gain|lag|resistance>       (one sub-axis at D1)
    perception.<noise|bias|latency|dropout>  (one sub-axis at D1)
    dynamic                    (D7 intruder at D1, admitted under the evaluation rules)
    appearing                  (appearing obstacle at D1, where the suite is wired)

Training doses (owner decision J10, 2026-10-06; v2.6 assembly G31): the light training dose of every suite is
its CURRENT D1 (TRAINING_LEVEL); the appearing D1 is 5.0 m for both boats. Where D1 does not exist by design
the hazard suite trains at D2 instead: BlueBoat station keeping, dynamic (G23e: the D1 intruder would have to
appear before the start) and appearing (G20d: the 10 m approach has no room for the 5.0 m gap). The table is
TRAINING_LEVEL_EXCEPTIONS; ``training_level(task, vessel, suite)`` / ``TrainingMixture.level(suite)`` is the
one place every caller reads the dose from (stack tokens, hazard gates, stamp).

No combinations: at most one component is active in an episode.

Mechanism (smallest change to the condition layer). The training env is built
with ONE resolved stack that holds every mixture suite at D1
(``required_tokens``; admissible on every journal task, CONTEXT_R4 D4). A
per-env, per-episode GATE then switches each axis on or off at reset:
``HydroV2ConditionLayer.attach_training_mixture`` makes the layer call
``gates(env_ids, episodes)`` at every reset and apply the gates in its
physics hooks (current, waves, wind), in ``configure_runtime`` (actuator
sub-axes) and in the perception degrader (perception sub-axes). A gated-off
axis contributes exactly what the disabled suite contributes. The hazard
suites (dynamic, appearing) are gated by the training hazard pools through
``hazard_dose`` (IMPLEMENTATION_PLAN.md item P3); the layer ignores them.

Assignment rule (block-balanced, deterministic). For env ``e`` the episodes
are cut into consecutive blocks of ``B = calm_slots + S`` episodes, S = the
number of mixture suites wired on the task (7 with appearing, else 6). Each
block holds every suite exactly once and ``calm`` exactly ``calm_slots``
times, in a keyed random order (Fisher-Yates over a SplitMix64 stream keyed by
``scenario_rng.stable_stream_seed(protocol, seed, env, block, "train.mix/1")``).
The sub-axis of the k-th actuator (perception) episode of an env cycles
through the four sub-axes in keyed random order per cycle of four. The
component of (seed, env, episode) is therefore a pure function of the key:
independent of the policy, of episode lengths, of reset order, of other envs
and of every global RNG; realised shares are exact per block.

Namespace rule. A mixture may only be built for a TRAINING seed (purpose
'training' or, for the training smoke, 'training_smoke' in the seed registry,
TRAINING_PURPOSES); a layer with a mixture attached refuses to produce a
certificate stamp (training never produces evaluation evidence).

D1 values (owner decision O2 pending: the D1 presets are not frozen here). The
mixture references every D1 preset BY NAME (``preset_names``) and stamps its
numeric values (``preset_records``: presets_v3.provenance for the layer suites;
the D7 TTC ratio and the appearing gap ratio of dose 1 for the hazard suites),
so the digest changes whenever a D1 value changes.

Calm share (owner decision O3 pending): ``calm_slots`` calm episodes per block,
default DEFAULT_CALM_SLOTS = 3 (30 % where appearing is wired, 3/9 elsewhere).

Pure Python / NumPy; imports no simulator package.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json

import numpy as np

try:
    from ..scenario_rng import SCENARIO_PROTOCOL_VERSION, stable_stream_seed
except ImportError:  # loaded outside the package (CPU tools)
    from scenario_rng import SCENARIO_PROTOCOL_VERSION, stable_stream_seed  # type: ignore

from .presets_v3 import JOURNAL_PRESETS, provenance

MIXTURE_SCHEMA = "usvbench.training_mixture/1"
SAMPLER_PROTOCOL = "usvbench.training_mixture.block_balanced/1"
STREAM_GROUP = "train.mix/1"
AXIS_STREAM_GROUP = "train.mix.axis/1"
TRAINING_LEVEL = 1              # owner decision J10: each suite's current D1 ...
# ... except where D1 is NOT APPLICABLE BY DESIGN (owner decision J10, 2026-10-06): (task, vessel, suite) -> level
TRAINING_LEVEL_EXCEPTIONS = {
    ("station_keeping", "blueboat_v2", "dynamic"): 2,     # G23e: BlueBoat SK D7 offers D2-D4 only
    ("station_keeping", "blueboat_v2", "appearing"): 2,   # G20d: BlueBoat SK appearing offers D2-D4 only
    ("docking", "blueboat_v2", "appearing"): 2,           # G58: BlueBoat docking appearing D1 not applicable by
}                                                         # design (appearing/ladder.py NOT_APPLICABLE_DOSES, 2026-10-14)
TRAINING_LEVEL_DECISION = ("owner decision J10 (2026-10-06): PPO training light dose = each suite's current D1 "
                           "(appearing D1 = 5.0 m both boats); BlueBoat station keeping dynamic and appearing "
                           "have no D1 by design and train at D2")
DEFAULT_CALM_SLOTS = 3          # owner decision O3 (pending): calm episodes per block
TRAINING_PURPOSES = ("training", "training_smoke")     # seed-registry purposes a mixture may use (G19b, P7)

ENVIRONMENT_SUITES = ("waves", "wind", "current")
AXIS_SUITES = {"actuator": ("cap", "gain", "lag", "resistance"),
               "perception": ("noise", "bias", "latency", "dropout")}
HAZARD_SUITES = ("dynamic", "appearing")
SUITE_ORDER = ("waves", "wind", "current", "actuator", "perception", "dynamic", "appearing")
# Gate names (one per axis the layer or a hazard pool switches)
GATES = ("waves", "wind", "current",
         "actuator.cap", "actuator.gain", "actuator.lag", "actuator.resistance",
         "perception.noise", "perception.bias", "perception.latency", "perception.dropout",
         "dynamic", "appearing")
COMPONENTS = ("calm",) + GATES

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)


class MixtureError(ValueError):
    """A training mixture was misconfigured or used outside training."""


def training_level(task, vessel, suite):
    """Training dose of ``suite`` on (task, vessel): TRAINING_LEVEL (D1) unless TRAINING_LEVEL_EXCEPTIONS
    names the combination (owner decision J10). Layer suites (waves .. perception) are always D1."""
    if suite not in SUITE_ORDER:
        raise MixtureError(f"{suite!r} is not a mixture suite")
    return int(TRAINING_LEVEL_EXCEPTIONS.get((str(task), str(vessel), str(suite)), TRAINING_LEVEL))


def _splitmix(key, start, count):
    """Elements ``start .. start + count - 1`` of the SplitMix64 stream at ``key`` (keyed_rng.py protocol)."""
    with np.errstate(over="ignore"):
        counters = np.arange(start + 1, start + count + 1, dtype=np.uint64)
        x = np.uint64(key) + counters * _GOLDEN
        z = (x ^ (x >> np.uint64(30))) * _MIX1
        z = (z ^ (z >> np.uint64(27))) * _MIX2
        return z ^ (z >> np.uint64(31))


def keyed_permutation(items, key):
    """Fisher-Yates permutation of ``items`` driven by the SplitMix64 stream at ``key``.

    Index draws use an exact integer reduction (unbiased by rejection), so the
    result depends only on the key and the item order.
    """
    out = list(items)
    n = len(out)
    if n < 2:
        return out
    buffer, used = [], 0

    def next_u64():
        nonlocal buffer, used
        if used == len(buffer):
            buffer = buffer + [int(v) for v in _splitmix(key, len(buffer), 64)]
        value = buffer[used]
        used += 1
        return value

    for i in range(n - 1, 0, -1):
        bound = i + 1
        limit = (1 << 64) - ((1 << 64) % bound)
        while True:
            value = next_u64()
            if value < limit:
                break
        j = value % bound
        out[i], out[j] = out[j], out[i]
    return out


@dataclass(frozen=True)
class TrainingMixture:
    """Block-balanced D1 mixture of one training run (one task, one vessel, one training seed)."""

    seed: int
    task: str
    vessel: str
    appearing_wired: bool
    calm_slots: int = DEFAULT_CALM_SLOTS
    purpose: str = "training"
    protocol_version: int = SCENARIO_PROTOCOL_VERSION

    def __post_init__(self):
        if self.purpose not in TRAINING_PURPOSES:
            raise MixtureError("a training mixture needs a TRAINING seed (registry purpose 'training' or "
                               f"'training_smoke'); got purpose {self.purpose!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise MixtureError("the mixture seed must be an integer training seed")
        if isinstance(self.calm_slots, bool) or not isinstance(self.calm_slots, int) or self.calm_slots < 0:
            raise MixtureError("calm_slots must be a nonnegative integer")
        if not isinstance(self.task, str) or not self.task or not isinstance(self.vessel, str) or not self.vessel:
            raise MixtureError("task and vessel are required")

    # ------------------------------------------------------------ structure --
    @property
    def suites(self):
        return tuple(s for s in SUITE_ORDER if s != "appearing" or self.appearing_wired)

    def level(self, suite):
        """Training dose of ``suite`` in this run (owner decision J10; see training_level)."""
        return training_level(self.task, self.vessel, suite)

    def training_levels(self):
        """{suite: training dose} of every suite of this mixture (stamped in spec, so it enters the digest)."""
        return {suite: self.level(suite) for suite in self.suites}

    @property
    def block_length(self):
        return self.calm_slots + len(self.suites)

    def block_slots(self):
        return ("calm",) * self.calm_slots + self.suites

    def weights(self):
        """Exact share of every component (Fractions; zero for absent components)."""
        b = self.block_length
        out = {name: Fraction(0) for name in COMPONENTS}
        out["calm"] = Fraction(self.calm_slots, b)
        for suite in self.suites:
            if suite in AXIS_SUITES:
                for axis in AXIS_SUITES[suite]:
                    out[f"{suite}.{axis}"] = Fraction(1, b * len(AXIS_SUITES[suite]))
            else:
                out[suite] = Fraction(1, b)
        return out

    def required_tokens(self):
        """The ONE condition stack the training env is built with (every mixture suite at its training dose:
        D1, or the J10 exception)."""
        tokens = [f"{suite}=D{self.level(suite)}" for suite in self.suites]
        return tokens

    def required_levels(self):
        """Expanded level keys a resolved stack must carry (journal_config levels form)."""
        levels = {}
        for suite in self.suites:
            if suite in AXIS_SUITES:
                for axis in AXIS_SUITES[suite]:
                    levels[f"{suite}.{axis}"] = self.level(suite)
            else:
                levels[suite] = self.level(suite)
        return levels

    # ------------------------------------------------------------- sampling --
    def _key(self, env, index, group):
        return stable_stream_seed(int(self.protocol_version), int(self.seed), int(env), int(index), group)

    def component(self, env, episode):
        """Component name of (env, episode); a pure function of (seed, env, episode)."""
        env, episode = int(env), int(episode)
        if env < 0 or episode < 0:
            raise MixtureError("env and episode indices must be nonnegative")
        block, position = divmod(episode, self.block_length)
        order = keyed_permutation(self.block_slots(), self._key(env, block, STREAM_GROUP))
        suite = order[position]
        if suite not in AXIS_SUITES:
            return suite
        axes = AXIS_SUITES[suite]
        cycle, slot = divmod(block, len(axes))      # each suite occurs exactly once per block
        axis_order = keyed_permutation(axes, self._key(env, cycle, AXIS_STREAM_GROUP + "." + suite))
        return f"{suite}.{axis_order[slot]}"

    def components(self, env_ids, episodes):
        ids = [int(i) for i in np.asarray(env_ids).reshape(-1)]
        eps = [int(e) for e in np.asarray(episodes).reshape(-1)]
        if len(ids) != len(eps):
            raise MixtureError("one episode index per env is required")
        return [self.component(i, e) for i, e in zip(ids, eps)]

    def gates(self, env_ids, episodes):
        """{gate: bool ndarray [len(env_ids)]}: True where that axis is ON for the episode."""
        names = self.components(env_ids, episodes)
        out = {gate: np.zeros(len(names), dtype=bool) for gate in GATES}
        for row, name in enumerate(names):
            if name != "calm":
                out[name][row] = True
        return out

    def hazard_dose(self, suite, env, episode):
        """Dose the training hazard pool must serve for (env, episode): the suite's training dose (D1, or the
        J10 exception) on its component, else D0."""
        if suite not in HAZARD_SUITES:
            raise MixtureError(f"{suite!r} is not a hazard suite")
        if suite == "appearing" and not self.appearing_wired:
            return 0
        return self.level(suite) if self.component(env, episode) == suite else 0

    # ---------------------------------------------------------------- stamp --
    def preset_records(self):
        """Provenance (numeric values included) of every D1 preset in training."""
        records = {}
        for suite in self.suites:
            if suite in AXIS_SUITES:
                for axis in AXIS_SUITES[suite]:
                    name = f"journal.{suite}.{axis}.D{TRAINING_LEVEL}"
                    records[f"{suite}.{axis}"] = provenance(name)
            elif suite in ENVIRONMENT_SUITES:
                records[suite] = provenance(f"journal.{suite}.D{TRAINING_LEVEL}")
            else:
                level = self.level(suite)
                records[suite] = dict(_hazard_d1_values(suite, self.vessel, level), suite=suite, level=level,
                                      rule=("admitted under the evaluation admission rules of the suite at D%d; "
                                            "a not-admitted key runs D0 (CONTEXT_R4 D16)" % level))
        return records

    def preset_names(self):
        """Names of the D1 presets in training (layer suites; the hazard suites are named by their dose)."""
        names = []
        for suite in self.suites:
            if suite in AXIS_SUITES:
                names += [f"journal.{suite}.{axis}.D{TRAINING_LEVEL}" for axis in AXIS_SUITES[suite]]
            else:
                names.append(f"journal.{suite}.D{TRAINING_LEVEL}" if suite in ENVIRONMENT_SUITES
                             else f"{suite}=D{self.level(suite)}")
        return names

    def spec(self):
        return {"schema": MIXTURE_SCHEMA, "sampler": SAMPLER_PROTOCOL,
                "stream_groups": [STREAM_GROUP, AXIS_STREAM_GROUP],
                "scenario_protocol_version": int(self.protocol_version),
                "seed": int(self.seed), "purpose": self.purpose, "task": self.task, "vessel": self.vessel,
                "training_level": TRAINING_LEVEL, "training_levels": self.training_levels(),
                "training_level_decision": TRAINING_LEVEL_DECISION, "calm_slots": self.calm_slots,
                "block_length": self.block_length, "suites": list(self.suites),
                "appearing_wired": bool(self.appearing_wired),
                "combinations": "none (at most one component per episode)",
                "weights": {k: str(v) for k, v in self.weights().items()},
                "required_tokens": self.required_tokens(),
                "preset_names": self.preset_names(),
                "presets": self.preset_records()}

    def digest(self):
        text = json.dumps(self.spec(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(text.encode("ascii")).hexdigest()

    def stamp(self, counts=None):
        out = deepcopy(self.spec())
        out["mixture_sha256"] = self.digest()
        if counts is not None:
            out["realised_episode_starts"] = {k: int(v) for k, v in sorted(counts.items())}
        return out


_T_AVOID = {}


def _t_avoid_s(vessel):
    """D7 envelope t_avoid of a vessel, built exactly as journal_env_mixin.journal_setup builds the envelope."""
    if vessel not in _T_AVOID:
        from .. import journal_cfg_v2
        from ..journal_vehicle import envelope_spec, resolve_journal_vehicle
        from .dynamic import envelope_from_spec_v2
        record = journal_cfg_v2.vessel_record(vessel)
        envelope, _ = envelope_from_spec_v2(envelope_spec(resolve_journal_vehicle(vessel)),
                                            reference_speed_mps=record.cruise_speed_m_s,
                                            turning_radius_m=record.turning_radius_design_m)
        _T_AVOID[vessel] = float(envelope.t_avoid_s)
    return _T_AVOID[vessel]


def _appearing_ladder_identity():
    """The active appearing ladder of THIS tree (suite version, option, kind, witnessed doses, gap table)."""
    from .appearing import config as ac
    summary = ac.ladder_summary() if callable(getattr(ac, "ladder_summary", None)) else {
        "ladder_option": None, "ladder_kind": "k", "gap_ratios": list(ac.GAP_RATIOS), "gaps_m": None}
    return {"suite_version": ac.APPEARING_SUITE_VERSION,
            "ladder_option": getattr(ac, "LADDER_OPTION", None), "ladder_kind": getattr(ac, "LADDER_KIND", "k"),
            "witnessed_doses": [int(d) for d in getattr(ac, "WITNESSED_DOSES", (1, 2, 3, 4))],
            "ladder": json.loads(json.dumps(summary, sort_keys=True))}


_VESSEL_PARAMETERS = ("vessel_id", "vessel", "vehicle")


def _trigger_gap_d1(ac, t_avoid, vessel, level=TRAINING_LEVEL):
    """trigger_gap_m(level, ...) (D1 unless the J10 exception) under the signature of THIS tree, detected by
    parameter names (no exception fallback): v2.5 trigger_gap_m(dose, t_avoid_s); G20b trigger_gap_m(dose,
    t_avoid_s, vessel_id=None)."""
    import inspect
    names = list(inspect.signature(ac.trigger_gap_m).parameters)
    if names[:2] != ["dose", "t_avoid_s"]:
        raise MixtureError("appearing trigger_gap_m%s: unknown signature (want dose, t_avoid_s[, vessel_id])"
                           % (tuple(names),))
    extra = names[2:]
    if not extra:
        return ac.trigger_gap_m(int(level), t_avoid)
    if len(extra) == 1 and extra[0] in _VESSEL_PARAMETERS:
        return ac.trigger_gap_m(int(level), t_avoid, **{extra[0]: vessel})
    raise MixtureError("appearing trigger_gap_m%s: unknown signature (want dose, t_avoid_s[, vessel_id])"
                       % (tuple(names),))


def _hazard_d1_values(suite, vessel, level=TRAINING_LEVEL):
    """Numeric definition of the training dose of a hazard suite (G19b, O2 stamp, follow-up M3; ``level`` = D1
    or the J10 exception, G31).

    dynamic: the D7 TTC ratio of dose 1 (dynamic/model.py DOSE_TTC_RATIOS[1]).
    appearing: the trigger gap of dose 1 in metres for this vessel from conditions/appearing/config.trigger_gap_m
    (k ladder: k_1 x t_avoid x u_ref; G20b metre ladder: the vessel's table), its equivalent k, and the identity of
    the active ladder with the witnessed doses (the training admission is joint over them), so the mixture digest
    changes whenever the appearing ladder or its D1 value changes."""
    if suite == "dynamic":
        try:
            from .dynamic.model import DOSE_TTC_RATIOS
        except ImportError:  # loaded outside the package (CPU tools)
            from dynamic.model import DOSE_TTC_RATIOS  # type: ignore
        return {"values": {"ttc_over_t_avoid": float(DOSE_TTC_RATIOS[int(level)])},
                "source": "tasks/_shared/conditions/dynamic/model.py DOSE_TTC_RATIOS[%d]" % int(level)}
    from .appearing import config as ac
    t_avoid = _t_avoid_s(vessel)
    gap = _trigger_gap_d1(ac, t_avoid, vessel, level)
    u_ref = float(ac.REFERENCE_SPEED_MPS)
    return {"values": {"trigger_gap_m": float(gap), "k_equivalent": float(gap) / (t_avoid * u_ref),
                       "t_avoid_s": t_avoid, "u_ref_mps": u_ref},
            "ladder": _appearing_ladder_identity(),
            "source": "tasks/_shared/conditions/appearing/config.py trigger_gap_m(%d, t_avoid, vessel) of this tree"
                      % int(level)}


def check_stack(mixture, resolved):
    """Fail closed unless ``resolved`` (journal_config.JournalConditions) is the mixture's stack."""
    levels = dict(getattr(resolved, "levels", ()))
    want = mixture.required_levels()
    have = {k: v for k, v in levels.items() if k not in ("dynamic", "appearing")}
    want_layer = {k: v for k, v in want.items() if k not in ("dynamic", "appearing")}
    if have != want_layer:
        raise MixtureError(f"training stack levels {sorted(have.items())} differ from the mixture's "
                           f"{sorted(want_layer.items())}")
    dyn_level = mixture.level("dynamic") if "dynamic" in mixture.suites else 0
    if bool(getattr(resolved, "dynamic_enabled", False)) != ("dynamic" in mixture.suites) or \
            int(getattr(resolved, "dynamic_dose", 0)) != dyn_level:
        raise MixtureError("the training stack must carry dynamic=D%d (owner decision J10)" % dyn_level)
    if bool(getattr(resolved, "appearing_enabled", False)) != mixture.appearing_wired:
        raise MixtureError("the training stack carries appearing=D%d exactly when the suite is wired"
                           % mixture.level("appearing"))
    if mixture.appearing_wired and int(getattr(resolved, "appearing_dose", 0)) != mixture.level("appearing"):
        raise MixtureError("the training stack must carry appearing=D%d (owner decision J10)"
                           % mixture.level("appearing"))
    if getattr(resolved, "overrides", ()):
        raise MixtureError("the training stack takes no parameter overrides")
    if int(resolved.seed) != int(mixture.seed):
        raise MixtureError("the training stack seed differs from the mixture seed")
    return True


def check_layer_config(mixture, conditions_cfg):
    """Fail closed unless a schema-3 config holds every mixture layer suite at its D1 preset."""
    for suite in ENVIRONMENT_SUITES:
        values = JOURNAL_PRESETS[f"journal.{suite}.D{TRAINING_LEVEL}"]["values"][suite]
        block = getattr(conditions_cfg, suite)
        for field, value in values.items():
            if field in ("direction_mode",):
                continue
            if getattr(block, field) != value:
                raise MixtureError(f"training layer {suite}.{field}={getattr(block, field)!r} is not the D1 "
                                   f"preset value {value!r}")
    for suite, axes in AXIS_SUITES.items():
        block = getattr(conditions_cfg, suite)
        if not block.enabled:
            raise MixtureError(f"training layer {suite} is not enabled")
        for axis in axes:
            values = JOURNAL_PRESETS[f"journal.{suite}.{axis}.D{TRAINING_LEVEL}"]["values"][suite]
            for field, value in values.items():
                if field == "enabled":
                    continue
                if getattr(block, field) != value:
                    raise MixtureError(f"training layer {suite}.{field}={getattr(block, field)!r} is not the "
                                       f"D1 preset value {value!r}")
    for field in ("vel_noise_sigma_mps", "yaw_rate_noise_sigma_rad_s", "ray_noise_sigma_m"):
        if getattr(conditions_cfg.perception, field) != 0.0:
            raise MixtureError("the sensor bundle is not a training component (perception." + field + ")")
    return True


__all__ = ["TrainingMixture", "MixtureError", "MIXTURE_SCHEMA", "SAMPLER_PROTOCOL", "GATES", "COMPONENTS",
           "TRAINING_LEVEL", "TRAINING_LEVEL_EXCEPTIONS", "TRAINING_LEVEL_DECISION", "training_level",
           "DEFAULT_CALM_SLOTS", "TRAINING_PURPOSES", "check_stack", "check_layer_config",
           "keyed_permutation"]
