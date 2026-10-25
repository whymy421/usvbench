"""Fail-closed, CPU-only inputs for all v3 certificate analyses.

Legacy files are diagnostics, not analysis inputs. Pairing checks every recorded
primitive unless its name is explicitly declared as the treatment. No collector
intersection, duplicate overwrite, metric coercion, or guessed deadline is used.
"""
from __future__ import annotations

import math
from pathlib import Path

from cert_v3 import EVALUATION_PROTOCOL_VERSION
import d7_exposure
import appearing_exposure
from validate_cert_v3 import PAIRING_SCHEMA, conditions_modules, load_json, validate_certificate


class AnalysisInputError(ValueError):
    """A certificate cannot support the requested scientific analysis."""


NUMERIC_METRICS = {
    "tts_s", "path_length_m", "path_length_success_prefix_m",
    "path_length_full_episode_m", "min_clearance_m", "contact_seconds",
    "contact_longest_seconds", "contact_depth_mean_m", "d0_m",
    "route_geodesic_m", "max_hold_s", "mean_cross_track_error_m",
}
INTEGER_METRICS = {"contact_steps", "max_phase", "gates_passed"}


def finite_number(value, label, *, nullable=False, positive=False):
    if value is None and nullable:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise AnalysisInputError(label + " must be a finite number, not a coerced value")
    if positive and value <= 0:
        raise AnalysisInputError(label + " must be positive")
    return float(value)


def validate_record_metrics(record, label="record"):
    if not isinstance(record, dict) or type(record.get("success")) is not bool:
        raise AnalysisInputError(label + ".success must be a boolean")
    for field, value in record.items():
        if field in NUMERIC_METRICS or field.endswith(("_m", "_s", "_mps", "_rad")):
            # Timing objects have their own schema, not a scalar metric type.
            if field in ("controller_compute_time_s", "environment_step_time_s"):
                continue
            finite_number(value, label + "." + field, nullable=True)
        elif field in INTEGER_METRICS and value is not None:
            if type(value) is not int or value < 0:
                raise AnalysisInputError(label + "." + field + " must be a nonnegative integer")


def validate_analysis_input(document, label="certificate", allowed_purposes=None):
    """Validate a complete v3 grid and return the unchanged input document.

    allowed_purposes, when given, restricts the certified seed purpose
    (e.g. ("final_test",) for main-table analyses, ("selection",) for the PPO
    competence gate). Diagnostic runs have purpose 'diagnostic'.
    """
    if not isinstance(document, dict) or document.get("schema_version") != 3:
        raise AnalysisInputError(label + ": schema v3 required; legacy input is not accepted")
    header = document.get("header")
    if not isinstance(header, dict):
        raise AnalysisInputError(label + ": header must be an object")
    if header.get("evaluation_protocol_version") != EVALUATION_PROTOCOL_VERSION:
        raise AnalysisInputError(label + ": evaluation_protocol_version %r required (fixed grid, "
                                 "seed namespace, resolved deadline)" % EVALUATION_PROTOCOL_VERSION)
    if allowed_purposes is not None:
        registration = header.get("seed_namespace")
        purpose = registration.get("purpose") if isinstance(registration, dict) else None
        if purpose not in tuple(allowed_purposes):
            raise AnalysisInputError("%s: seed purpose %r is not one of %s"
                                     % (label, purpose, ", ".join(allowed_purposes)))
    n, quota, requested = (header.get(k) for k in
                            ("num_envs", "episodes_per_env", "episodes_requested"))
    if any(type(v) is not int or v <= 0 for v in (n, quota, requested)):
        raise AnalysisInputError(label + ": positive integer grid dimensions required")
    if requested != n * quota:
        raise AnalysisInputError(label + ": episode budget must equal the complete divisible grid")
    errors = validate_certificate(document, num_envs=n, episodes_per_env=quota)
    if errors:
        raise AnalysisInputError(label + ": " + "; ".join(errors))
    for index, record in enumerate(document["records"]):
        validate_record_metrics(record, label + f".records[{index}]")
        # JSON Schema's integer accepts integral floats, but episode IDs must
        # be typed integers, and booleans must never alias integer keys.
        if type(record["env"]) is not int or type(record["ep"]) is not int:
            raise AnalysisInputError(label + ": env and ep must be typed integers")
    return document


def load_analysis_input(path):
    return validate_analysis_input(load_json(Path(path)), str(path))


def resolved_deadline(document):
    """Read the resolved deadline; C5 cfg snapshot is the sole legacy-v3 fallback."""
    header = document["header"]
    if "resolved_deadline_s" in header:
        value = header["resolved_deadline_s"]
    else:
        value = header.get("resolved_config", {}).get("episode_length_s")
    return finite_number(value, "resolved episode deadline", positive=True)


def speed_reference(document):
    """Return the vessel-labelled reference or None when explicitly unavailable."""
    header = document["header"]
    reference = header.get("speed_reference")
    if reference is None:
        return None
    if (not isinstance(reference, dict)
            or not {"vessel", "value_m_s", "basis"} <= set(reference)
            or not set(reference) <= {"vessel", "value_m_s", "basis", "source"}):
        raise AnalysisInputError("speed_reference needs vessel, value_m_s, basis (and optional source)")
    if not isinstance(reference["vessel"], str) or reference["vessel"] != header["vessel"]:
        raise AnalysisInputError("speed_reference vessel differs from certificate vessel")
    if reference["basis"] not in ("nominal", "condition_adjusted"):
        raise AnalysisInputError("speed_reference basis must be 'nominal' or 'condition_adjusted'")
    finite_number(reference["value_m_s"], "speed_reference.value_m_s", positive=True)
    return reference


def _protocol(document):
    return document["header"].get("scenario_protocol", document.get("scenario_protocol"))


def validate_compatible(documents, same_scenario_protocol=False):
    """Require the same science/software release, while permitting host/GPU changes.

    same_scenario_protocol additionally requires an identical scenario protocol
    stamp (paired analyses); unpaired per-cell tables may mix families that are
    on and off the protocol.
    """
    docs = list(documents)
    for doc in docs:
        validate_analysis_input(doc)
    if not docs:
        raise AnalysisInputError("no certificates supplied")
    first = docs[0]
    for other in docs[1:]:
        a, b = first["header"], other["header"]
        for field in ("physics_version", "hydrodynamics_model", "protocol_version",
                      "evaluation_protocol_version", "analysis_protocol_version"):
            if a.get(field) != b.get(field):
                raise AnalysisInputError("incompatible " + field)
        # E1: the release code tree (usvbench-code-tree-v1) is the code identity;
        # the C5 directory hashes are compared only between legacy blocks.
        if a["code"]["hash_algorithm"] != b["code"]["hash_algorithm"]:
            raise AnalysisInputError("incompatible code.hash_algorithm")
        fields = (("code_tree_sha256",) if a["code"]["hash_algorithm"] == "usvbench-code-tree-v1"
                  else ("tasks_sha256", "scripts_sha256"))
        for field in fields:
            if a["code"][field] is None or a["code"][field] != b["code"][field]:
                raise AnalysisInputError("incompatible code." + field)
        for field in ("isaac_sim_version", "isaac_lab_version", "physx_version"):
            if a["runtime"][field] != b["runtime"][field]:
                raise AnalysisInputError("incompatible runtime." + field)
        if same_scenario_protocol and _protocol(first) != _protocol(other):
            raise AnalysisInputError("incompatible scenario_protocol")


def is_journal_certificate(header):
    """F7 / D11: a journal certificate declares its primitive inventory in header.pairing."""
    declaration = header.get("pairing")
    return isinstance(declaration, dict) and declaration.get("schema") == PAIRING_SCHEMA


def _required_primitives(header, declared):
    """Known v2 family/config minima, plus any explicit protocol declaration.

    A shared missing channel is not evidence of pairing. Unknown task families
    must supply a primitive declaration. New active stochastic suites need their
    own hashes; legacy perception draws without hashes deliberately fail closed.

    F7: a journal certificate (header.pairing schema usvbench.pairing/1,
    written by journal_env_mixin.journal_certificate_header) requires EVERY
    primitive of its declared inventory (condition primitives, d7_assignment
    and the family primitives); the task-name heuristics below are legacy only.
    """
    if is_journal_certificate(header):
        primitives = header["pairing"].get("primitives")
        if not isinstance(primitives, list) or not primitives:
            raise AnalysisInputError("journal pairing declaration without a primitive inventory")
        return set(primitives) | set(declared)
    task, cfg = header["task_id"].lower(), header["resolved_config"]
    required = set(declared)
    if "stationkeep" in task:
        required.add("spawn_pose")
    elif "pathhazard" in task:
        required.update(("spawn_pose", "layout"))
    elif "pathfollow" in task:
        required.update(("spawn_pose", "route"))
    elif "harbor" in task:
        required.add("layout")
        if cfg.get("spawn_phase_jitter_m", 0) > 0:
            required.add("spawn_pose")
    elif "hazard" in task or "iceberg" in task:
        required.update(("layout", "rotation"))
    elif not declared:
        raise AnalysisInputError("unknown task primitive inventory: supply header.pairing declaration")
    sea = cfg.get("sea_state")
    if isinstance(sea, dict) and sea.get("enable", False):
        required.add("wave")
    physics = cfg.get("underwater_physics_cfg")
    if isinstance(physics, dict) and physics.get("enable_current", False):
        required.add("current")
    if cfg.get("appear_count", 0) > 0:
        required.add("appearance")
    for name in ("thrust_imbalance", "mass_scale", "drag_scale", "thrust_cap_scale", "motor_tau_s", "payload_mass_kg"):
        if cfg.get(name + "_choices"):
            required.add("actuator_" + name)
    if any(value not in (0, 0.0, None) for key, value in cfg.items()
           if key.startswith(("obs_noise_sigma_", "obs_bias_sigma_")) or key == "obs_dropout_p"):
        required.add("perception")
    return required


# journal treatment primitive -> schema-3 condition block it lives in
_SUITE_BLOCK = {"waves": "waves", "wind": "wind", "current": "current"}


def _suite_block(primitive):
    if primitive in _SUITE_BLOCK:
        return _SUITE_BLOCK[primitive]
    head = primitive.split(".", 1)[0]
    return head if head in ("actuator", "perception") else None


def dynamic_exposure(record):
    """D16: "admitted" or "not_admitted" of a journal record's D7 stamp."""
    try:
        return d7_exposure.exposure_of(record["dynamic_hazards"])[0]
    except (KeyError, d7_exposure.ExposureError) as exc:
        raise AnalysisInputError("record has no readable D7 exposure: %s" % exc) from exc


def appearing_record_exposure(record):
    """G3: "admitted" / "not_admitted" of a journal record's appearing stamp (digest verified), or None."""
    envelope = record.get("appearing_hazards")
    if envelope is None:
        return None
    try:
        return appearing_exposure.exposure_of(envelope)[0]
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisInputError("record has no readable appearing exposure: %s" % exc) from exc


def _appearing_unexposed(pairing, a, b, treatment, changed):
    """The appearing analogue of the D16/D27 rule (G3 phase 1b).

    Under the appearing treatment a pair in which either side's appearing stamp is
    not admitted is UNEXPOSED: the key never received a cylinder (it ran D0, its
    stamp says assigned false, itt_count 0) and its appearing primitive is the
    dose-independent not-admitted digest, identical in both certificates. Every
    stored hash is still recomputed from its stamp, the scenario identity and the
    context must match and no primitive outside the treatment may differ; the
    pair is coverage, never a treated pair and never a zero difference in the
    effect. Returns "unexposed", or None when the pair is not such a pair.
    """
    if getattr(pairing, "APPEARING_PRIMITIVE", None) not in changed:
        return None
    exposures = (appearing_record_exposure(a), appearing_record_exposure(b))
    if appearing_exposure.NOT_ADMITTED not in exposures:
        return None
    hashes = [pairing._checked_hashes(a, "baseline"), pairing._checked_hashes(b, "treatment")]
    if set(hashes[0]) != set(hashes[1]):
        raise pairing.PairingError(f"primitive inventories differ: {sorted(set(hashes[0]) ^ set(hashes[1]))}")
    if pairing._identity(a) != pairing._identity(b):
        raise pairing.PairingError("different scenarios: %s vs %s" % (pairing._identity(a), pairing._identity(b)))
    ca, cb = a["conditions"]["context"], b["conditions"]["context"]
    for field in _PAIR_CONTEXT:
        if ca.get(field) != cb.get(field):
            raise pairing.PairingError(f"context differs outside the treatment: {field}")
    outside = sorted(n for n in hashes[0] if hashes[0][n] != hashes[1][n] and n not in changed)
    if outside:
        raise pairing.PairingError(f"unpaired difference outside the treatment {treatment!r}: {outside}")
    return "unexposed"


_PAIR_CONTEXT = ("vehicle", "family", "observation_contract", "physics_dt_s", "control_dt_s",
                 "hydrodynamics_model", "vessel_profile")


def _not_admitted_pair(pairing, a, b, treatment, changed):
    """pairing.validate_pair semantics for a record pair with a not-admitted side (D16).

    Every stored hash is recomputed (condition primitives from the schema-3
    stamps, d7_assignment by d7_exposure.d7_primitive); the scenario identity and
    context must match; no primitive outside the treatment may differ. Under the
    dynamic treatment the pair is UNEXPOSED (the not-admitted side ran D0, no
    intruder; its stamp says assigned false, itt_count 0, CONTEXT_R4 D27) and is
    returned as such: coverage, never a treated pair and never a zero difference
    in the effect; under any other treatment the D7 primitive must be identical
    on both sides (the key is not admitted in both certificates) and the
    treatment primitives must differ. Returns "unexposed" or "paired".
    """
    hashes = []
    for label, record in (("baseline", a), ("treatment", b)):
        stored = record.get("scenario_hashes")
        if not isinstance(stored, dict):
            raise pairing.PairingError(label + ": scenario_hashes required")
        recomputed = pairing.condition_primitive_hashes(record["conditions"])
        recomputed[pairing.D7_PRIMITIVE] = d7_exposure.d7_primitive(pairing, record["dynamic_hazards"])
        for name, value in recomputed.items():
            if stored.get(name) != value:
                raise pairing.PairingError(f"{label}: scenario hash {name!r} does not match its stamp")
        hashes.append(stored)
    ha, hb = hashes
    if set(ha) != set(hb):
        raise pairing.PairingError(f"primitive inventories differ: {sorted(set(ha) ^ set(hb))}")

    def identity(record):
        rt = record["conditions"]["runtime"]
        return (record["conditions"]["parameters"]["seed"], rt["env_index"], rt["episode_index"],
                rt["scenario_ordinal"])

    if identity(a) != identity(b):
        raise pairing.PairingError(f"different scenarios: {identity(a)} vs {identity(b)}")
    ca, cb = a["conditions"]["context"], b["conditions"]["context"]
    for field in _PAIR_CONTEXT:
        if ca.get(field) != cb.get(field):
            raise pairing.PairingError(f"context differs outside the treatment: {field}")
    differing = sorted(name for name in ha if ha[name] != hb[name])
    outside = [name for name in differing if name not in changed]
    if outside:
        raise pairing.PairingError(f"unpaired difference outside the treatment {treatment!r}: {outside}")
    if pairing.D7_PRIMITIVE in changed:
        return "unexposed"
    if not differing:
        raise pairing.PairingError(f"no treatment difference: every primitive of {treatment!r} is identical")
    return "paired"


def validate_journal_pair_detailed(baseline, treatment, changed_primitives=()):
    """validate_journal_pair plus the D16 exposure of every pair.

    Returns {"pairs": [(a, b)] (every validated pair of the grid), "status":
    ["paired" | "unexposed"], "effect_pairs": [(a, b)] (the pairs with status
    "paired"), "unexposed": n, "not_admitted": {"baseline": n, "treatment": n},
    "dynamic_treatment": bool}.
    CONTEXT_R4 D27: under the dynamic treatment the effect is estimated over the
    admitted-key pairs only (intention to treat within the ASSIGNED population:
    a not-admitted key is never assigned, its stamp says assigned false,
    itt_count 0); a pair with a not-admitted side is "unexposed" and is reported
    as coverage ("unexposed", "not_admitted"), never as treated and never as a
    zero difference added to the effect's denominator, so it is not in
    "effect_pairs". Under every other treatment not-admitted records pair like
    any other record ("effect_pairs" == "pairs").
    """
    return _validate_journal_pair(baseline, treatment, changed_primitives)


def validate_journal_pair(baseline, treatment, changed_primitives=()):
    """Journal pairing (D11) with conditions/pairing.py semantics.

    ``changed_primitives``: the declared treatment as suite names or primitive
    names (pairing.SUITE_PRIMITIVES keys, e.g. ("waves",) or ("actuator.cap",));
    when empty the certified header declaration is used. Every record pair
    must be the same scenario (seed, env, episode, scenario ordinal), every
    primitive outside the treatment must be identical, every stored hash must
    match its stamp, and the treatment primitives must differ
    (pairing.validate_pair per record). The header condition parameters may
    differ only in the treated suite blocks (plus the preset list), and the D7
    header only when the treatment is the dynamic suite.

    Round 4b (D16): records whose D7 stamp is an explicit not_admitted record
    pair with the same key of the other certificate; for every treatment other
    than the dynamic suite they pair normally. CONTEXT_R4 D27: for the dynamic
    treatment they are unexposed coverage and are NOT returned (the returned
    pairs are the effect population: admitted-key pairs only);
    validate_journal_pair_detailed reports the coverage.
    """
    return _validate_journal_pair(baseline, treatment, changed_primitives)["effect_pairs"]


def _validate_journal_pair(baseline, treatment, changed_primitives=()):
    _, pairing, _ = conditions_modules()
    validate_compatible((baseline, treatment), same_scenario_protocol=False)
    ah, bh = baseline["header"], treatment["header"]
    if not (is_journal_certificate(ah) and is_journal_certificate(bh)):
        raise AnalysisInputError("a journal certificate pairs only with a journal certificate")
    for field in ("task_id", "vessel", "level", "eval_seed", "num_envs", "episodes_per_env",
                  "seed_namespace", "journal", "vessel_profile"):
        if ah.get(field) != bh.get(field):
            raise AnalysisInputError("paired header mismatch: " + field)
    if ah["pairing"].get("primitives") != bh["pairing"].get("primitives"):
        raise AnalysisInputError("paired primitive inventories differ")
    declared = [tuple(h["pairing"].get("changed_primitives") or ()) for h in (ah, bh)]
    if declared[0] != declared[1]:
        raise AnalysisInputError("paired primitive declarations differ")
    names = tuple(changed_primitives) or declared[0]
    try:
        changed = set(pairing.treatment_primitives(names))
    except pairing.PairingError as exc:
        raise AnalysisInputError(str(exc)) from exc
    if declared[0] and changed != set(declared[0]):
        raise AnalysisInputError("caller treatment differs from certified pairing declaration")
    if not changed <= set(ah["pairing"]["primitives"]):
        raise AnalysisInputError("treatment primitives are not in the certified inventory")
    blocks = {_suite_block(p) for p in changed} - {None}
    pa, pb = ah["conditions"]["parameters"], bh["conditions"]["parameters"]
    for key in sorted(set(pa) | set(pb)):
        if key in blocks or key == "presets":
            continue
        if pa.get(key) != pb.get(key):
            raise AnalysisInputError("header condition parameters differ outside the treatment: " + key)
    dynamic = pairing.D7_PRIMITIVE in changed
    da, db = ah.get("dynamic_hazards_header"), bh.get("dynamic_hazards_header")
    if not dynamic and da != db:
        raise AnalysisInputError("dynamic-hazard headers differ outside the treatment")
    # G3: the appearing header may differ only in its dose (and the suite block that names it) under the
    # appearing treatment; it is identical otherwise
    appearing = getattr(pairing, "APPEARING_PRIMITIVE", None) in changed
    xa, xb = ah.get("appearing_hazards_header"), bh.get("appearing_hazards_header")
    if (xa is None) != (xb is None):
        raise AnalysisInputError("one certificate carries the appearing suite and the other does not")
    if xa is not None:
        strip = ("dose", "suite") if appearing else ()
        if {k: v for k, v in xa.items() if k not in strip} != {k: v for k, v in xb.items() if k not in strip}:
            raise AnalysisInputError("appearing headers differ outside the treatment")
    treatment_names = names if len(names) != 1 else names[0]
    ar = {(r["env"], r["ep"]): r for r in baseline["records"]}
    br = {(r["env"], r["ep"]): r for r in treatment["records"]}
    if set(ar) != set(br):
        raise AnalysisInputError("paired grids differ; intersections are not analysis populations")
    pairs, status = [], []
    counts = {"baseline": 0, "treatment": 0}
    appearing_counts = {"baseline": 0, "treatment": 0}
    for key in sorted(ar):
        a, b = ar[key], br[key]
        try:
            ea, eb = dynamic_exposure(a), dynamic_exposure(b)
            counts["baseline"] += ea == d7_exposure.NOT_ADMITTED
            counts["treatment"] += eb == d7_exposure.NOT_ADMITTED
            appearing_counts["baseline"] += appearing_record_exposure(a) == appearing_exposure.NOT_ADMITTED
            appearing_counts["treatment"] += appearing_record_exposure(b) == appearing_exposure.NOT_ADMITTED
            label = _appearing_unexposed(pairing, a, b, treatment_names, changed)    # G3 (D27 analogue)
            if label is not None:
                status.append(label)
            elif ea == eb == d7_exposure.ADMITTED:
                pairing.validate_pair(a, b, treatment_names)
                status.append("paired")
            else:
                status.append(_not_admitted_pair(pairing, a, b, treatment_names, changed))
        except (pairing.PairingError, KeyError, TypeError, ValueError) as exc:
            raise AnalysisInputError(f"{key}: {exc}") from exc
        pairs.append((a, b))
    effect = [pair for pair, label in zip(pairs, status) if label == "paired"]     # D27
    return {"pairs": pairs, "status": status, "effect_pairs": effect, "unexposed": status.count("unexposed"),
            "not_admitted": counts, "dynamic_treatment": dynamic,
            "appearing_not_admitted": appearing_counts, "appearing_treatment": appearing}


def validate_pair(baseline, treatment, changed_primitives=()):
    """Return sorted complete row pairs after checking every unchanged primitive.

    ``changed_primitives`` is an explicit treatment declaration, never a list
    inferred from observed mismatches. Optional header.pairing declarations
    must agree across certificates and cannot be contradicted by the caller.
    Journal certificates (header.pairing schema usvbench.pairing/1) are
    paired by validate_journal_pair (conditions/pairing.py semantics).
    """
    if is_journal_certificate(baseline.get("header", {})) or is_journal_certificate(
            treatment.get("header", {})):
        return validate_journal_pair(baseline, treatment, changed_primitives)
    validate_compatible((baseline, treatment), same_scenario_protocol=True)
    ah, bh = baseline["header"], treatment["header"]
    for field in ("task_id", "vessel", "level", "eval_seed", "num_envs", "episodes_per_env",
                  "seed_namespace"):
        if ah.get(field) != bh.get(field):
            raise AnalysisInputError("paired header mismatch: " + field)
    protocol = _protocol(baseline)
    if not isinstance(protocol, dict) or not protocol.get("version"):
        raise AnalysisInputError("pairing requires a declared active scenario_protocol version")
    changed = set(changed_primitives)
    if any(not isinstance(x, str) or not x for x in changed):
        raise AnalysisInputError("changed primitive names must be nonempty strings")
    if "scenario" in changed:
        raise AnalysisInputError("scenario is a derived aggregate, not a treatment primitive")
    declaration = ah.get("pairing")
    if declaration != bh.get("pairing"):
        raise AnalysisInputError("paired primitive declarations differ")
    unchanged_declared = set()
    if declaration is not None:
        if not isinstance(declaration, dict):
            raise AnalysisInputError("pairing declaration must be an object")
        for key in ("unchanged_primitives", "changed_primitives"):
            values = declaration.get(key, [])
            if (not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values)
                    or len(set(values)) != len(values)):
                raise AnalysisInputError("invalid pairing." + key)
        unchanged_declared = set(declaration.get("unchanged_primitives", []))
        declared_changed = set(declaration.get("changed_primitives", []))
        if changed and changed != declared_changed:
            raise AnalysisInputError("caller treatment differs from certified pairing declaration")
        changed = declared_changed
        if changed & unchanged_declared:
            raise AnalysisInputError("a primitive cannot be both changed and unchanged")
    required = (_required_primitives(ah, changed | unchanged_declared)
                | _required_primitives(bh, changed | unchanged_declared))
    ar = {(r["env"], r["ep"]): r for r in baseline["records"]}
    br = {(r["env"], r["ep"]): r for r in treatment["records"]}
    if set(ar) != set(br):
        raise AnalysisInputError("paired grids differ; intersections are not analysis populations")
    inventory = None
    pairs = []
    for key in sorted(ar):
        a, b = ar[key], br[key]
        x, y = a.get("scenario_hashes"), b.get("scenario_hashes")
        if not isinstance(x, dict) or not isinstance(y, dict) or not x or not y:
            raise AnalysisInputError(f"{key}: complete primitive hash maps are required")
        if set(x) != set(y):
            raise AnalysisInputError(f"{key}: primitive inventories differ")
        if inventory is None:
            inventory = set(x)
        if set(x) != inventory:
            raise AnalysisInputError(f"{key}: primitive inventory changes within the grid")
        # A treatment primitive is not compared, so it need not be hashed
        # (e.g. perception noise has no scenario hash yet). Every unchanged
        # required or declared primitive must be present on both sides.
        if not ((required - changed) | unchanged_declared) <= set(x):
            raise AnalysisInputError(f"{key}: declared primitive is missing")
        if not (set(x) - changed - {"scenario"}):
            raise AnalysisInputError("pairing requires at least one unchanged primitive")
        for primitive in x:
            if any(not isinstance(v, str) or not v for v in (x[primitive], y[primitive])):
                raise AnalysisInputError(f"{key}: invalid primitive hash {primitive}")
            # This digest hashes the whole scenario, including treatment values.
            # It must match only when no primitive is intentionally changed.
            compare = primitive not in changed and not (primitive == "scenario" and changed)
            if compare and x[primitive] != y[primitive]:
                raise AnalysisInputError(f"{key}: unchanged primitive differs: {primitive}")
        pairs.append((a, b))
    return pairs


def validate_metric(document, field):
    """Validate a selected scalar analysis endpoint; null means unavailable."""
    for record in document["records"]:
        finite_number(record.get(field), "metric " + field, nullable=True)
