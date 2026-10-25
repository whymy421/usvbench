"""Arbitrary suite stacking for journal configurations (CONTEXT_R4 D4; F0).

ONE entry point, ``resolve_journal_conditions(presets=..., overrides=...,
seed=...)``, accepts any subset of the six suites at independently chosen
levels, with the same syntax from the command line and from a JSON/YAML file:

    --suite waves=D2 wind=D3 actuator.cap=D1 dynamic=D2 wind.offset_deg=30

Level tokens (``<key>=D<k>``, k = 0..4; appearing also D5 under ladder M, diagnostic runs only):
    waves | waves.A      ladder A (Hs 0/0.125/0.25/0.5/1.0 m, Tp 4 s)
    waves.B              ladder B (CEM fetch-limited, U10 0/3/6/9/12 m/s)
    wind                 U10 0/3/6/9/12 m/s (D0 = module off)
    current              0/0.25/0.5/1.0/1.5 m/s
    actuator             shorthand: all four sub-axes at the same level
    actuator.cap | .gain | .lag | .resistance
    perception           shorthand: the four primary axes at the same level
    perception.noise | .bias | .latency | .dropout | .bundle   (bundle: labelled
                         sensor bundle, velocity + range noise, never implied)
    dynamic              D7 dose (TTC/t_avoid 4/3/2/1.25; D0 = no intruder)
    appearing            G3 appearing obstacle: a static cylinder appears when the
                         own-ship footprint first comes within the trigger gap d_k
                         of its surface, d_k from the active ladder option
                         (appearing/ladder.py; default M = per-vessel metres,
                         owner decision 2026-10-03: BlueBoat 5.0/2.9/1.7/1.0 m,
                         WAM-V 5.0/3.6/2.5/1.8 m, + supplementary appearing=D5 =
                         the witnessed calibration rung 0.76/1.25 m, diagnostic
                         runs only; k ladders A 4/3/2/1.25, B 4/2/1/0.5 for
                         reference, d_k = k x t_avoid x u_ref; D0 = never
                         appears). Wired on every journal
                         task (appearing/config.WIRED_TASKS, G20b). Diagnostic
                         only: appearing.gap_m=<x> overrides the trigger gap (m) of
                         the run (calibration sweep, x from D5 up to the task's
                         start gap; refused in official runs).
Parameter tokens (``<suite>.<field>=<value>``):
    <suite>.offset_deg   waves/current/wind: offset from the shared compass-8
                         stratum; perception.offset_deg: bias direction offset
    <suite>.<field>      any schema-3 field (config_v3) or DynamicSuiteConfig
                         field, e.g. current.speed_mps=0.3, wind.gust_intensity=0.2
                         (the appearing suite takes no parameter: its admission
                         fields are pinned to the admitted pool)
Values: D<k> for levels; JSON literals (true, 0.3, [[0, 0], [60, 0.5]]) or
bare strings for parameters. A preset name (``journal.waves.D2``, a
``journal.combo.*`` name) is accepted as a level token too.
Supplementary presets (G10): ``journal.waves.supp075`` (BlueBoat boundary
level Hs 0.75 m, Tp 4 s, ladder-A rules; a SUPPLEMENT, never a main-table
level) is selected by its name exactly like a combination preset; it owns the
waves slot (C1 against waves=D<k>) and is kept in ``tokens()`` and in the
stamp's ``supplement`` list.
Combination presets (G10b): a ``journal.combo.*`` name expands to its
component presets exactly as presets_v3.expand_presets does (each component
owns its slot, C1 against a different level for that slot; the components
share the scenario keys and the compass-8 stratum, COMBINATION_RULE), is kept
by name in ``tokens()`` (so the canonical tokens re-resolve to the identical
stack) and is stamped by name with its provenance (role, rule, components) in
the stamp's ``combinations`` list.

Declared conflicts (the ONLY rejections; each error names the reason):
  C1 two values for one field (e.g. actuator=D2 with actuator.cap=D1, or
     waves=D2 with waves.B=D3, or a level and a parameter token that set the
     same field to different values);
  C2 an offset or a dose field on a suite/axis that is not enabled (it would
     silently do nothing);
  C3 an offset with a fixed world direction (two values for one direction),
     or an offset outside (-180, 180] deg;
  C4 a perception dose on a measurement the task's observation contract does
     not consume (observation_groups.admit_perception; e.g. the sensor bundle's
     velocity noise on path following, whose journal layout has no velocity);
  C5 physically inadmissible values rejected by schema 3 (e.g. a wave dose
     failing DNV-RP-C205 steepness, latency/outage not a multiple of the
     control period, a wave band the hull needs raised with band_policy fixed);
  C6 unknown suite, axis, field or level.
  C7 the appearing suite on a task where it is not wired (G20b: none; the
     check stays as a guard for a task outside appearing/config.WIRED_TASKS).
Everything else stacks: every subset of {waves, wind, current, actuator (4),
perception (4 + bundle), dynamic} at independent levels, on every task and
both vessels (CPU proof: scripts/tests_journal_glue/test_stacking.py), and
every subset of those seven suites including appearing on the wired tasks
(scripts/tests_journal_glue/test_stacking_appearing.py).

Directional suites share the scenario's compass-8 stratum; the offsets are
stamped in the schema-3 parameters (hence in the parameter digest).
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, fields
import json
import math
import re
from pathlib import Path

from .config_v3 import SUITES_V3, JournalConditionConfig, resolve_v3
from .presets_v3 import COMBINATIONS, JOURNAL_PRESETS, LEVELS, SUPPLEMENT_PRESETS, expand_presets, provenance
from .stamp import parameters_digest

SCHEMA = "usvbench.journal_conditions/1"
SUITES = ("waves", "wind", "current", "actuator", "perception", "dynamic", "appearing")
AXES = {"actuator": ("cap", "gain", "lag", "resistance"),
        "perception": ("noise", "bias", "latency", "dropout", "bundle")}
SHORTHAND_AXES = {"actuator": AXES["actuator"], "perception": ("noise", "bias", "latency", "dropout")}
_LEVEL = re.compile(r"^D([0-4])$")
OFFSET_FIELD = {"waves": "direction_offset_rad", "current": "direction_offset_rad",
                "wind": "direction_offset_rad", "perception": "bias_direction_offset_rad"}
DECLARED_CONFLICTS = {
    "C1": "two values for one field",
    "C2": "offset or dose on a suite/axis that is not enabled",
    "C3": "offset with a fixed world direction, or outside (-180, 180] deg",
    "C4": "perception dose on a measurement the observation contract does not consume",
    "C5": "physically inadmissible value (schema-3 admission)",
    "C6": "unknown suite, axis, field or level",
    "C7": "suite not wired on this task (appearing obstacle, G3 phase 1)",
}
# dose fields per suite: an override of one of them needs the suite enabled (C2)
_NEUTRAL = {
    "waves": {"hs_m": 0.0}, "current": {"speed_mps": 0.0, "speed_table": ()},
    "wind": {"u10_mps": 0.0},
    "actuator": {"thrust_cap": 1.0, "motor_gain_one_sided": 1.0, "additional_lag_s": 0.0,
                 "resistance_scale": 1.0},
    "perception": {"pos_noise_sigma_m": 0.0, "pos_bias_m": 0.0, "latency_s": 0.0, "outage_s": 0.0,
                   "vel_noise_sigma_mps": 0.0, "yaw_rate_noise_sigma_rad_s": 0.0, "ray_noise_sigma_m": 0.0},
}


class SuiteConflict(ValueError):
    """A declared conflict (see DECLARED_CONFLICTS); ``code`` names it."""

    def __init__(self, code, message):
        super().__init__(f"declared conflict {code} ({DECLARED_CONFLICTS[code]}): {message}")
        self.code = code


def _dynamic_module():
    from . import dynamic
    return dynamic


def _appearing_config_module():
    from .appearing import config
    return config


# ------------------------------------------------------------------ parsing --
def _parse_value(text):
    if not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except ValueError:
        return text


def _appearing_supplementary_level(value):
    """G20b: the level of an appearing token D5.. that the ACTIVE ladder declares (option E), else None."""
    match = re.match(r"^D([0-9]+)$", str(value).strip())
    if not match or int(match.group(1)) <= 4:
        return None
    level = int(match.group(1))
    return level if level in _appearing_config_module().SUPPLEMENTARY_DOSES else None


def _level(value, key):
    if isinstance(value, int) and not isinstance(value, bool) and value in LEVELS:
        return int(value)
    match = _LEVEL.match(str(value).strip())
    if not match:
        raise SuiteConflict("C6", f"{key}: level must be D0..D4, got {value!r}")
    return int(match.group(1))


def _tokens_from(items):
    """Normalise tokens / mappings / preset names into ['key=value', ...]."""
    if items is None:
        return []
    if isinstance(items, str):
        items = [items]
    if isinstance(items, dict):
        out = []
        for key, value in items.items():
            if isinstance(value, dict):
                out.extend(f"{key}.{field}={json.dumps(v) if not isinstance(v, str) else v}"
                           for field, v in value.items())
            else:
                out.append(f"{key}={json.dumps(value) if not isinstance(value, str) else value}")
        return out
    return [str(item) for item in items]


def parse_suite_tokens(tokens):
    """Split tokens into (levels {key: k}, params {(suite, field): value}, preset names)."""
    levels, params, names = {}, {}, []
    for token in _tokens_from(tokens):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            if token in JOURNAL_PRESETS or token in COMBINATIONS:
                names.append(token)
                continue
            raise SuiteConflict("C6", f"token {token!r} is neither key=value nor a journal preset name")
        key, value = (part.strip() for part in token.split("=", 1))
        suite = key.split(".", 1)[0]
        if suite not in SUITES:
            raise SuiteConflict("C6", f"unknown suite {suite!r} in {token!r}; suites: {SUITES}")
        tail = key.split(".", 1)[1] if "." in key else None
        if suite == "appearing" and tail is None and _appearing_supplementary_level(value) is not None:
            level = _appearing_supplementary_level(value)      # G20b: ladder option E's supplementary D5
            if key in levels and levels[key] != level:
                raise SuiteConflict("C1", f"{key} given twice ({levels[key]} and {value})")
            levels[key] = level
            continue
        if re.match(r"^D[0-9]+$", value) and not _LEVEL.match(value):
            raise SuiteConflict("C6", f"{key}: level must be D0..D4, got {value!r}")
        is_level = bool(_LEVEL.match(value))
        if is_level:
            if tail is not None and not (suite in AXES and tail in AXES[suite]) and not (
                    suite == "waves" and tail in ("A", "B")):
                raise SuiteConflict("C6", f"unknown axis {key!r}")
            if key in levels and levels[key] != _level(value, key):
                raise SuiteConflict("C1", f"{key} given twice ({levels[key]} and {value})")
            levels[key] = _level(value, key)
        else:
            if tail is None:
                raise SuiteConflict("C6", f"{token!r}: a suite takes a level D0..D4 (parameters are "
                                          "suite.field=value)")
            field = (suite, tail)
            parsed = _parse_value(value)
            if field in params and params[field] != parsed:
                raise SuiteConflict("C1", f"{key} given twice ({params[field]!r} and {parsed!r})")
            params[field] = parsed
    return levels, params, names


def _flatten_overrides(overrides):
    """Nested {suite: {field: v}} / dotted {'suite.field': v} / tokens -> params."""
    if overrides is None:
        return {}
    if isinstance(overrides, dict) and all(isinstance(v, dict) for v in overrides.values()):
        out = {}
        for suite, values in overrides.items():
            for field, value in values.items():
                out[(suite, field)] = value
        return out
    levels, params, names = parse_suite_tokens(overrides)
    if levels or names:
        raise SuiteConflict("C6", "overrides take parameters only; put levels in presets")
    return params


# ------------------------------------------------------------------ result --
@dataclass(frozen=True)
class JournalConditions:
    """Resolved stack: schema-3 conditions + the D7 dynamic level (+ the G3 appearing level)."""

    conditions: JournalConditionConfig
    seed: int
    levels: tuple                 # sorted ((key, level), ...), after shorthand expansion
    presets: tuple                # schema-3 preset names applied
    overrides: tuple              # sorted ((suite, field, value), ...) of explicit parameters
    dynamic_dose: int = 0
    dynamic_enabled: bool = False
    dynamic_overrides: tuple = ()
    appearing_dose: int = 0
    appearing_enabled: bool = False
    appearing_gap_m: object = None        # G20b diagnostic token appearing.gap_m (None: the ladder's gap)

    def appearing_config(self, vessel_id):
        """G3 AppearingSuiteConfig bound to the task vessel (registry name)."""
        return _appearing_config_module().AppearingSuiteConfig(
            enabled=self.appearing_enabled, dose=self.appearing_dose,
            vessel_id=str(vessel_id) if vessel_id else "", gap_override_m=self.appearing_gap_m)

    def dynamic_config(self, vessel_id):
        """E4 DynamicSuiteConfig bound to the task vessel (registry name)."""
        dynamic = _dynamic_module()
        values = dict(self.dynamic_overrides)
        values.update(enabled=self.dynamic_enabled, dose=self.dynamic_dose,
                      vessel_id=str(vessel_id) if vessel_id else "")
        return dynamic.DynamicSuiteConfig(**values)

    def tokens(self):
        """Canonical token list (re-resolves to an identical result)."""
        out = [f"{key}=D{level}" for key, level in self.levels]
        # G10 / G10b: supplementary and combination presets are selected by name (no level key), so they
        # are kept here by name, in their resolution order
        out += [name for name in self.presets if name in SUPPLEMENT_PRESETS or name in COMBINATIONS]
        out += [f"{suite}.{field}={json.dumps(value)}" for suite, field, value in self.overrides]
        if self.appearing_gap_m is not None:          # G20b: absent token -> the token list is unchanged
            out.append(f"appearing.gap_m={json.dumps(self.appearing_gap_m)}")
        return out

    def stamp(self):
        """JSON-native provenance block with the schema-3 parameter digest."""
        parameters = json.loads(json.dumps(asdict(self.conditions), sort_keys=True))
        out = {"schema": SCHEMA, "seed": self.seed, "tokens": self.tokens(),
               "levels": {key: level for key, level in self.levels},
               "presets": list(self.presets),
               "conditions_parameters_sha256": parameters_digest(parameters),
               "dynamic": {"enabled": self.dynamic_enabled, "dose": self.dynamic_dose,
                           "overrides": {k: v for k, v in self.dynamic_overrides}},
               "direction_rule": "shared compass-8 stratum; per-suite offsets stamped in parameters"}
        supplement = [name for name in self.presets if name in SUPPLEMENT_PRESETS]
        if supplement:                      # G10: absent supplement -> the stamp is unchanged (byte-identical)
            out["supplement"] = {"presets": supplement, "role": "SUPPLEMENT", "main_table": False}
        combinations = [name for name in self.presets if name in COMBINATIONS]
        if combinations:                    # G10b: absent combination -> the stamp is unchanged (byte-identical)
            out["combinations"] = [provenance(name) for name in combinations]
        if self.appearing_enabled:          # absent suite: the stamp is unchanged (byte-identical)
            config = _appearing_config_module()
            out["appearing"] = {"enabled": True, "dose": self.appearing_dose,
                                "suite_version": config.APPEARING_SUITE_VERSION,
                                "reference_speed_mps": config.REFERENCE_SPEED_MPS, **config.ladder_summary()}
            if self.appearing_gap_m is not None:      # G20b diagnostic token (refused in official runs)
                out["appearing"].update(gap_override_m=self.appearing_gap_m, diagnostic_only=True)
        return out


def _preset_names(levels):
    """Expand level keys into schema-3 preset names, checking C1 per slot.

    A slot (waves, wind, current, actuator.<axis>, perception.<axis>) may be
    named several times only with the SAME preset (e.g. actuator=D2 plus
    actuator.cap=D2); two different presets for one slot are a C1 conflict.
    """
    owners, sources, names = {}, {}, []

    def claim(slot, preset, source):
        if slot in owners and owners[slot] != preset:
            raise SuiteConflict("C1", f"{slot} is set by {sources[slot]} and by {source}")
        if slot not in owners:
            owners[slot], sources[slot] = preset, source
            names.append(preset)

    dynamic = None
    for key, level in sorted(levels.items()):
        suite, _, tail = key.partition(".")
        source = f"{key}=D{level}"
        if suite == "dynamic":
            dynamic = level
        elif suite == "appearing":
            continue                      # G3: no schema-3 preset; resolved separately
        elif suite == "waves":
            claim("waves", f"journal.waves.B.D{level}" if tail == "B" else f"journal.waves.D{level}", source)
        elif suite in ("wind", "current"):
            claim(suite, f"journal.{suite}.D{level}", source)
        else:
            for axis in ((tail,) if tail else SHORTHAND_AXES[suite]):
                claim(f"{suite}.{axis}", f"journal.{suite}.{axis}.D{level}", source)
    return names, dynamic, owners


def resolve_journal_conditions(presets=(), overrides=None, *, seed):
    """Resolve any stack of suites at independent levels (D4). See module doc.

    presets: tokens ("waves=D2", ...), a mapping {"waves": "D2", ...} or
        preset names; parameter tokens ("wind.offset_deg=30") may be mixed in.
    overrides: parameters as a nested mapping, dotted mapping or tokens.
    seed: the evaluation seed (mandatory integer; CONTEXT_R4 D7).
    """
    if seed is None or isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("resolve_journal_conditions needs an integer evaluation seed (CONTEXT_R4 D7)")
    levels, params, raw_names = parse_suite_tokens(presets)
    for field, value in _flatten_overrides(overrides).items():
        if field in params and params[field] != value:
            raise SuiteConflict("C1", f"{field[0]}.{field[1]} given twice")
        params[field] = value
    names, dynamic_level, owners = _preset_names(levels)
    for name in raw_names:
        for component, values in expand_presets((name,)):
            for suite in values:
                slot = suite if suite in ("waves", "wind", "current") else \
                    f"{suite}.{JOURNAL_PRESETS[component]['axis']}"
                if slot in owners and owners[slot] != component:
                    raise SuiteConflict("C1", f"{slot} is set by {owners[slot]} and by {name}")
                owners[slot] = component
        if name not in names:
            names.append(name)
    # preset-owned field values, for C1 against explicit parameters
    preset_fields = {}
    for component, values in expand_presets(tuple(names)):
        for suite, suite_values in values.items():
            for field, value in suite_values.items():
                if field != "enabled":
                    preset_fields[(suite, field)] = (component, value)
    nested, dynamic_overrides = {}, {}
    appearing_gap = None
    for (suite, field), value in sorted(params.items()):
        if suite == "appearing" and field == "gap_m":
            # G20b: diagnostic trigger-gap override (calibration sweep); the vessel bounds are checked when the
            # suite is bound to its vessel and task (AppearingSuiteConfig, AppearingBridge)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                    or value <= 0:
                raise SuiteConflict("C5", f"appearing.gap_m must be a positive number of metres, got {value!r}")
            appearing_gap = float(value)
            continue
        if suite == "appearing":
            raise SuiteConflict("C6", f"appearing.{field} is not an overridable field (use appearing=D<k> for "
                                      "the dose, appearing.gap_m=<x> for a diagnostic gap; the admission fields "
                                      "are pinned to the appearing pool)")
        if suite == "dynamic":
            known = {f.name for f in fields(_dynamic_module().DynamicSuiteConfig)}
            if field in ("enabled", "dose", "vessel_id") or field not in known:
                raise SuiteConflict("C6", f"dynamic.{field} is not an overridable D7 field "
                                          "(use dynamic=D<k> for the dose)")
            dynamic_overrides[field] = value
            continue
        if field == "offset_deg":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise SuiteConflict("C6", f"{suite}.offset_deg must be a finite number")
            if not -180.0 < float(value) <= 180.0:
                raise SuiteConflict("C3", f"{suite}.offset_deg={value} is outside (-180, 180]")
            field, value = OFFSET_FIELD[suite], math.radians(float(value))
        if suite not in SUITES_V3 or field not in {f.name for f in fields(SUITES_V3[suite])}:
            raise SuiteConflict("C6", f"unknown field {suite}.{field}")
        if isinstance(value, list):
            value = _tuple(value)
        owner = preset_fields.get((suite, field))
        if owner is not None and _tuple(owner[1]) != value:
            raise SuiteConflict("C1", f"{suite}.{field}: level {owner[0]} sets {owner[1]!r}, the parameter "
                                      f"sets {value!r}")
        nested.setdefault(suite, {})[field] = value
    try:
        conditions = resolve_v3(tuple(names), nested or None, seed=seed)
    except ValueError as error:
        text = str(error).replace("declared conflict: ", "")
        code = ("C2" if "is disabled" in text else "C3" if "direction_mode='strata'" in text or
                "(-pi, pi]" in text else "C6" if text.startswith("unknown") else "C5")
        raise SuiteConflict(code, text) from None
    for (suite, field) in ((s, f) for s, values in nested.items() for f in values):
        if suite in _NEUTRAL and field in _NEUTRAL[suite] and not getattr(conditions, suite).enabled:
            raise SuiteConflict("C2", f"{suite}.{field} is set but {suite} is not enabled (use a level or "
                                      f"{suite}.enabled=true)")
    level_items = dict(levels)
    for key, level in list(level_items.items()):
        suite, _, tail = key.partition(".")
        if suite in SHORTHAND_AXES and not tail:
            del level_items[key]
            for axis in SHORTHAND_AXES[suite]:
                level_items[f"{suite}.{axis}"] = level
        elif key == "waves.A":
            del level_items[key]
            level_items["waves"] = level
    dose = 0 if dynamic_level is None else dynamic_level
    appearing_level = levels.get("appearing")
    result = JournalConditions(
        conditions=conditions, seed=int(seed), levels=tuple(sorted(level_items.items())),
        presets=tuple(names),
        overrides=tuple(sorted((s, f, v) for s, values in nested.items() for f, v in values.items())),
        dynamic_dose=dose, dynamic_enabled=dynamic_level is not None,
        dynamic_overrides=tuple(sorted(dynamic_overrides.items())),
        appearing_dose=0 if appearing_level is None else int(appearing_level),
        appearing_enabled=appearing_level is not None, appearing_gap_m=appearing_gap)
    if appearing_gap is not None and not result.appearing_dose:
        raise SuiteConflict("C2", "appearing.gap_m is set but the appearing suite is not at a dose >= 1 "
                                  "(use e.g. appearing=D1 appearing.gap_m=1.4)")
    result.dynamic_config("validation_only")      # validates the D7 fields now (C5/C6)
    if result.appearing_enabled:
        result.appearing_config("validation_only")
    return result


def _tuple(value):
    if isinstance(value, list):
        return tuple(_tuple(v) for v in value)
    return value


def resolve_from_tokens(tokens, *, seed):
    """CLI/file form: one token list mixing levels and parameters."""
    return resolve_journal_conditions(presets=tokens, seed=seed)


# ------------------------------------------------------------ files and CLI --
def load_suite_file(path):
    """(tokens, seed or None) from a JSON or YAML suite file.

    Accepted layouts: {"suite": ["waves=D2", "wind.offset_deg=30"], "seed": 42}
    or {"suites": {"waves": "D2", "actuator.cap": "D1"},
        "overrides": {"wind": {"offset_deg": 30}}, "seed": 42}.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ValueError("YAML suite files need PyYAML; use the JSON form") from None
        data = yaml.safe_load(text)
    else:
        data = json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    if not isinstance(data, dict):
        raise ValueError("a suite file holds a mapping")
    unknown = set(data) - {"suite", "suites", "overrides", "seed"}
    if unknown:
        raise ValueError(f"unknown suite-file keys: {sorted(unknown)}")
    tokens = _tokens_from(data.get("suite")) + _tokens_from(data.get("suites"))
    tokens += _tokens_from(data.get("overrides"))
    seed = data.get("seed")
    return tokens, seed


def add_suite_arguments(parser):
    """--suite TOKEN... (repeatable) and --suite-file PATH on an argparse parser."""
    parser.add_argument("--suite", nargs="+", action="extend", default=[], metavar="TOKEN",
                        help="suite levels and parameters, e.g. waves=D2 wind=D3 actuator.cap=D1 "
                             "dynamic=D2 wind.offset_deg=30")
    parser.add_argument("--suite-file", default=None, metavar="PATH",
                        help="JSON/YAML file with the same tokens (see SUITES_HOWTO.md)")
    return parser


def resolve_from_args(args, *, seed=None):
    tokens = list(getattr(args, "suite", []) or [])
    file_seed = None
    if getattr(args, "suite_file", None):
        file_tokens, file_seed = load_suite_file(args.suite_file)
        tokens = file_tokens + tokens
    chosen = seed if seed is not None else file_seed
    if seed is not None and file_seed is not None and seed != file_seed:
        raise SuiteConflict("C1", f"seed {seed} (command line) and {file_seed} (suite file)")
    return resolve_from_tokens(tokens, seed=chosen)


def resolve_from_args_with_wind_coefficients(args, *, seed=None):
    """G16: ``resolve_from_args`` plus the diagnostic-only ``windcoef.*`` tokens.

    Returns (JournalConditions, override {name: float} or None). The windcoef tokens
    (command line or suite file) are split off BEFORE the stack is resolved, so the
    JournalConditions (tokens, stamp, parameter digest) is exactly the one of the same
    command without them; the caller (eval_protocol.apply_journal_suites) checks the
    run purpose and installs the override with ``apply_wind_coefficients``.
    """
    from .wind_coefficients import WindCoefficientError, split_tokens
    tokens = list(getattr(args, "suite", []) or [])
    file_seed = None
    if getattr(args, "suite_file", None):
        file_tokens, file_seed = load_suite_file(args.suite_file)
        tokens = file_tokens + tokens
    chosen = seed if seed is not None else file_seed
    if seed is not None and file_seed is not None and seed != file_seed:
        raise SuiteConflict("C1", f"seed {seed} (command line) and {file_seed} (suite file)")
    try:
        rest, override = split_tokens(tokens)
    except WindCoefficientError as exc:
        raise SuiteConflict("C1" if "C1" in str(exc) else "C6", str(exc)) from None
    return resolve_from_tokens(rest, seed=chosen), override


def apply_wind_coefficients(env_cfg, override):
    """G16: install a validated diagnostic override (``env_cfg.journal_wind_coefficients``).

    Called only for an override (never with None): a run at the pinned values keeps
    its cfg free of the attribute, so its certificate is unchanged.
    """
    from .wind_coefficients import validate_override
    override = validate_override(override)
    if override is None:
        raise ValueError("apply_wind_coefficients needs an override (pinned runs set nothing)")
    env_cfg.journal_wind_coefficients = dict(override)
    return env_cfg


def resolve_from_args_with_overrides(args, *, seed=None):
    """T25 (owner decision T1.35b): ``resolve_from_args`` plus BOTH diagnostic-only pseudo-suites,
    G16's ``windcoef.*`` and T25's ``hydro.rollpitch_zeta``.

    Returns (JournalConditions, wind override {name: float} or None, roll/pitch override
    {"rollpitch_zeta": float} or None). Both are split off BEFORE the stack is resolved, so the
    JournalConditions (tokens, stamp, parameter digest) is exactly the one of the same command
    without them; the caller (eval_protocol.apply_journal_suites) checks the run purpose and
    installs them with ``apply_wind_coefficients`` / ``apply_rollpitch_override``.
    """
    from .rollpitch_override import RollPitchOverrideError, split_tokens as split_rollpitch
    from .wind_coefficients import WindCoefficientError, split_tokens as split_wind
    tokens = list(getattr(args, "suite", []) or [])
    file_seed = None
    if getattr(args, "suite_file", None):
        file_tokens, file_seed = load_suite_file(args.suite_file)
        tokens = file_tokens + tokens
    chosen = seed if seed is not None else file_seed
    if seed is not None and file_seed is not None and seed != file_seed:
        raise SuiteConflict("C1", f"seed {seed} (command line) and {file_seed} (suite file)")
    try:
        rest, wind = split_wind(tokens)
        rest, rollpitch = split_rollpitch(rest)
    except (WindCoefficientError, RollPitchOverrideError) as exc:
        raise SuiteConflict("C1" if "C1" in str(exc) else "C6", str(exc)) from None
    return resolve_from_tokens(rest, seed=chosen), wind, rollpitch


def apply_rollpitch_override(env_cfg, override):
    """T25 (T1.35b): install a validated diagnostic roll/pitch override (``env_cfg.journal_rollpitch_zeta``).

    Called only for an override (never with None): a run without it keeps its cfg free of the
    attribute, so its physics and certificate are unchanged.
    """
    from .rollpitch_override import CFG_FIELD, NAME, validate_override
    override = validate_override(override)
    if override is None:
        raise ValueError("apply_rollpitch_override needs an override (runs without it set nothing)")
    setattr(env_cfg, CFG_FIELD, override[NAME])
    return env_cfg


def apply_journal_conditions(env_cfg, resolved):
    """Install a resolved stack on a journal env cfg (before the env is built)."""
    if not isinstance(resolved, JournalConditions):
        raise TypeError("apply_journal_conditions needs resolve_journal_conditions(...)")
    current = getattr(env_cfg, "seed", None)
    if current is not None and current != resolved.seed:
        raise ValueError(f"env cfg seed {current} differs from the resolved condition seed {resolved.seed}")
    env_cfg.seed = resolved.seed
    env_cfg.conditions = resolved.conditions
    env_cfg.condition_presets = ()
    env_cfg.journal_conditions = resolved
    return env_cfg


def journal_family(env_cfg):
    from .observation_groups import JOURNAL_TASK_CONTRACTS
    task = getattr(env_cfg, "journal_task", None)
    if task not in JOURNAL_TASK_CONTRACTS:
        raise ValueError(f"not a journal cfg (journal_task={task!r})")
    return JOURNAL_TASK_CONTRACTS[task][0]


_VEHICLE_CACHE = {}


def _journal_vehicle(name, options=None):
    # G4-A5: the cfg's journal_vehicle_options (e.g. swayyaw_damping) resolve the same
    # profile journal_env_mixin.journal_setup builds.
    options = dict(options or {})
    key = (name, tuple(sorted(options.items())))
    if key not in _VEHICLE_CACHE:
        from ..journal_vehicle import resolve_journal_vehicle
        _VEHICLE_CACHE[key] = resolve_journal_vehicle(name, **options)
    return _VEHICLE_CACHE[key]


def check_journal_cfg(env_cfg):
    """Frozen journal facts every journal env relies on (fail closed)."""
    from ..journal_cfg_v2 import CONTRACT_VERSION, PHYSICS_DT_S, PHYSICS_VERSION
    if getattr(env_cfg, "journal_contract_version", None) != CONTRACT_VERSION:
        raise ValueError("journal_contract_version must be 2")
    physics = getattr(env_cfg, "underwater_physics_cfg", None)
    if getattr(physics, "hydrodynamics_model", "legacy") != "hydro_v2":
        raise ValueError("journal cfgs select underwater_physics_cfg.hydrodynamics_model='hydro_v2' "
                         "(CONTEXT_R3 decision 13)")
    if int(getattr(physics, "physics_version", 1)) != PHYSICS_VERSION:
        raise ValueError("journal cfgs run physics_version=2")
    if not math.isclose(float(env_cfg.sim.dt), PHYSICS_DT_S, rel_tol=0, abs_tol=1e-15) or \
            int(env_cfg.decimation) != 2:
        raise ValueError("journal cfgs keep sim.dt = 1/120 s and decimation = 2 (CONTEXT_R4 D2)")
    return journal_family(env_cfg)


def admit_journal_conditions(env_cfg, resolved, *, num_envs=2, device="cpu"):
    """Resolve + ADMIT a stack on one journal cfg; returns a report or raises.

    Admission = the full schema-3 layer build on the cfg (contract, perception
    groups, legacy-overlap guards), binding to a real HydroV2Runtime of the
    cfg's vessel (wave band raised/checked against the hull, wind converted
    with the vessel windage), and the D7 suite bound to the vessel with the
    capacity/sensor preconditions. D7 encounter admission itself is offline
    per (key, dose) (D7 B05, E4 admit_grid) and is not repeated here.
    """
    import torch
    from ..hydro_v2_runtime import HydroV2Runtime
    from ..journal_cfg_v2 import D7_RESERVED_SLOTS, STATIC_CAPACITY
    from .layer import build_conditions
    family = check_journal_cfg(env_cfg)
    cfg = copy.copy(env_cfg)
    apply_journal_conditions(cfg, resolved)
    try:
        layer = build_conditions(cfg, num_envs, device, family=family)
    except ValueError as error:
        raise SuiteConflict(_admission_code(str(error)), str(error)) from None
    vehicle = _journal_vehicle(cfg.vehicle, getattr(cfg, "journal_vehicle_options", None))
    gravity = getattr(cfg.sim, "gravity", None)
    runtime = HydroV2Runtime(vehicle.hydro_profile, num_envs, torch.device(device), float(cfg.sim.dt),
                             engine_gravity=None if gravity is None else tuple(gravity))
    try:
        layer.bind_runtime(runtime)
    except ValueError as error:
        raise SuiteConflict("C5", str(error)) from None
    dynamic = resolved.dynamic_config(cfg.vehicle)
    if dynamic.enabled and "ray" not in layer.contract.groups and not dynamic.journal_ray_sensor_all_tasks:
        raise SuiteConflict("C4", "dynamic hazards on a task without a range sensor")
    task = cfg.journal_task
    if hasattr(cfg, "max_obstacles"):
        need = STATIC_CAPACITY[task] + D7_RESERVED_SLOTS[task]
        if int(cfg.max_obstacles) < need:
            raise ValueError(f"max_obstacles {cfg.max_obstacles} < static capacity + D7 slot = {need}")
    appearing = None
    if resolved.appearing_enabled:
        acfg = _appearing_config_module()
        if task not in acfg.WIRED_TASKS:
            raise SuiteConflict("C7", f"the appearing suite is wired on {', '.join(acfg.WIRED_TASKS)} only, "
                                      f"not on {task!r}")
        if "ray" not in layer.contract.groups:
            raise SuiteConflict("C4", "the appearing suite reaches controllers only through the range sensor")
        slot = acfg.appearing_slot(STATIC_CAPACITY[task], D7_RESERVED_SLOTS[task])
        width = getattr(cfg, "obstacle_slot_count", None)
        if width is not None and int(width) < slot + 1:
            raise ValueError(f"obstacle storage {width} has no appearing slot {slot} (capacity appearing_reserve)")
        appearing = asdict(resolved.appearing_config(cfg.vehicle))
        appearing["slot"] = slot
    stamp = layer.certificate_stamp()
    report = {"task": task, "vessel": cfg.vehicle, "family": family,
              "contract": layer.contract.contract, "levels": dict(resolved.levels),
              "parameters_sha256": stamp["parameters_sha256"],
              "resolved": stamp["context"]["resolved"], "dynamic": asdict(dynamic)}
    if appearing is not None:
        report["appearing"] = appearing
    return report


_C4_MARKERS = ("targets group", "does not consume", "velocity noise needs", "yaw-rate noise",
               "observation hooks are not wired")


def _admission_code(text):
    """C4 for a perception dose the contract cannot receive; C5 otherwise."""
    return "C4" if any(marker in text for marker in _C4_MARKERS) else "C5"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Resolve a journal suite stack (CONTEXT_R4 D4).")
    add_suite_arguments(parser)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    resolved = resolve_from_args(args, seed=args.seed)
    print(json.dumps(resolved.stamp(), indent=1, sort_keys=True))
    return resolved


__all__ = ["SCHEMA", "SUITES", "AXES", "DECLARED_CONFLICTS", "SuiteConflict", "JournalConditions",
           "parse_suite_tokens", "resolve_journal_conditions", "resolve_from_tokens", "load_suite_file",
           "add_suite_arguments", "resolve_from_args", "resolve_from_args_with_wind_coefficients",
           "apply_wind_coefficients", "resolve_from_args_with_overrides", "apply_rollpitch_override",
           "apply_journal_conditions", "check_journal_cfg",
           "admit_journal_conditions", "journal_family", "main"]
