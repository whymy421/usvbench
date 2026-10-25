"""Batch auditor: are two "independent" evaluation seeds really independent?

WHY THIS EXISTS
---------------
A certification run is quoted at two evaluation seeds (42 and 123) so that
agreement between the two numbers means something. That only holds if the two
seeds actually drew DIFFERENT episode scenarios. They did not, for whole task
families:

  * scripts/eval_v6_frozen.py:74 sets ``env_cfg.seed = args_cli.eval_seed``.
  * Isaac Lab's DirectRLEnv seeds the GLOBAL torch RNG from ``cfg.seed`` while
    the env is constructed.
  * The evaluator then builds an skrl Runner, and ``Runner.__init__`` calls
    ``set_seed(self._cfg.get("seed", None))`` with the constant pinned on line 1
    of e.g. tasks/station_keeping/agents/skrl_ppo_cfg.yaml -- which RESETS the
    global torch RNG and throws the evaluation seed away.
  * Families that draw their scenario from their OWN generator survive:
    tasks/hazard_nav/hazard_nav_env.py:96 builds
    ``np.random.default_rng(layout_seed)`` from ``cfg.seed``.
    Families that draw from the GLOBAL torch RNG do not: station keeping takes
    spawn distance / angle / heading with bare ``torch.rand``
    (tasks/station_keeping/station_keeping_env.py:684-688) and the shared sea
    state takes its wave phases the same way (tasks/_shared/sea_state.py:114).

So "seed 42 vs seed 123" can be one measurement reported twice.

WHAT THIS ADDS OVER scripts/check_eval_seed_independence.py
-----------------------------------------------------------
That script is the right idea on two files at a time. This one is the gate:

  1. Batch. Takes directories/globs, forms every seed pair itself, and exits
     non-zero if ANY pair is bad, so it can run in CI over gcerts*/.
  2. Aligns episodes by the ``(env, ep)`` key, not by list position. Records are
     appended in episode-completion order (eval_v6_frozen.py, ``records.append``
     inside the done-loop), so two runs that end episodes at different times
     produce records in a different ORDER and of different length -- real pairs
     in gcerts_cache/ share only 78-128 of their 128 keys. Positional zip
     therefore compares unrelated episodes.
  3. Separates SCENARIO channels from OUTCOME channels. ``d0_m`` and
     ``route_geodesic_m`` are latched from the layout at reset
     (tasks/hazard_nav/hazard_nav_env.py:1775-1777) and are therefore functions
     of the scenario ALONE, independent of the controller. ``tts_s``,
     ``path_length_m``, ``min_clearance_m`` and the ``contact_*`` family are
     outcomes: they move if EITHER the scenario or the controller changes.
     Conflating the two is how a partially-fixed family reads as clean.
  4. Catches the SUBTLE variant the success-pattern census cannot see: a family
     where some primitives are seed-derived and others are still
     short-circuited. That does not produce an identical success pattern, but it
     does leave some channel pinned while others move -> PARTIAL_SHORT_CIRCUIT.
  5. Weights every match by how surprising it is. A field that is constant
     within a run (``contact_steps`` is 0 for every episode of a clean run,
     ``min_clearance_m`` is a fixed constant in the docking certificates)
     matches across seeds for free and proves nothing. See "chance guard" below.
  6. Reads per-primitive scenario hashes when a certificate carries them.

SCENARIO-HASH SCHEMA (forward compatible)
-----------------------------------------
No certificate written so far carries scenario hashes, so the auditor degrades
to the numeric channels above. Once the evaluator stamps them, the canonical
shape this reads is:

    {
      "scenario_protocol": {"version": 2, "hash": "blake2b", ...},   # header
                            # "version" must be the CURRENT protocol version
                            # (SCENARIO_PROTOCOL_VERSION in
                            # tasks/_shared/scenario_rng.py); anything else is
                            # classified invalid/stale, never "on"
      "records": [
        {"env": 0, "ep": 0, ...,
         "scenario_hashes": {"spawn": "9f3c...", "current": "1a02...",
                             "wave": "77bd...", "actuator": "...",
                             "obs_degradation": "..."}},
        ...
      ]
    }

Accepted aliases, in precedence order:
    record["scenario_hashes"]  : dict primitive -> hex
    record["scenario_hash"]    : dict primitive -> hex, or a single hex string
                                 (recorded under the primitive name "scenario")
    cert["scenario_hashes"]    : dict primitive -> list aligned to records
                                 order, or dict primitive -> {"<env>:<ep>": hex}

The hash itself must be a STABLE hash of
``(protocol_version, eval_seed, env_index, episode_index, primitive_group)`` --
hashlib (blake2b/sha256) or numpy SeedSequence. Python's builtin ``hash()`` is
salted per process and would differ between two runs of the SAME scenario, which
would make this auditor report false independence.

HOW A PAIR IS CLASSIFIED
------------------------
Episodes are aligned on ``(env, ep)``; only the intersection is compared.
Every per-episode field present on both sides becomes a CHANNEL. For each
channel we compute, over the n episodes where at least one side has a value:

    frac   = fraction of episodes whose values are equal (exact by default;
             see --float-tol)
    q      = per-episode collision probability of the channel, taken from its
             own within-certificate value distribution, worst case of the two
             certificates: q = max_c sum_v (count_c(v)/n)^2.  This is the
             probability that two INDEPENDENT episodes of that certificate
             carry the same value.
    chance = q**n, the probability that two independent runs match on ALL n
             episodes by luck.

Channel status:
    IDENTICAL           frac == 1.0 and chance <= --chance-threshold
                        -> a match that cannot be luck: evidence of SAMENESS.
    UNINFORMATIVE_MATCH frac == 1.0 but chance > --chance-threshold
                        -> e.g. all-fail success, or contact_steps == 0
                        everywhere. Proves nothing; carries no weight.
    ANOMALOUS           frac < 1.0 but frac > q + 3*sqrt(q(1-q)/n) + 1/n
                        -> more agreement than chance allows, less than total:
                        the signature of partial sharing.
    DIFFERING           anything else -> evidence of DIFFERENCE. A mismatch is
                        always informative: evaluation is deterministic
                        (eval_v6_frozen.py takes ``mean_actions``), so identical
                        scenarios would reproduce identical numbers.
    SKIPPED             no comparable episodes.

Scenario-hash channels bypass the chance guard: a cryptographic hash cannot
collide by luck, so any hash equality is real evidence. A primitive stamped by
only ONE certificate of a pair is SKIPPED and warned about rather than compared
against a missing value -- a schema gap must not read as independence.

THE OFF-PROTOCOL MARKER IS AN OUTCOME, NOT A FOOTNOTE
-----------------------------------------------------
An evaluator whose env never built a ScenarioRNG stamps the literal
``"off-protocol"`` into ``scenario_protocol`` instead of the header block
(tasks/_shared/scenario_draws.py:349, scripts/eval_v6_frozen.py:91). That
certificate's episodes are NOT keyed by (protocol version, eval seed, env index,
episode index, group): they come from a generator advanced by resets, so on a
family that ends episodes early two runs reach reset k after consuming different
numbers of draws and the (env, ep) keys this auditor aligns on do not name the
same scenario on both sides. Every channel comparison below then rests on a
false premise, in EITHER direction -- a DIFFERING channel may be two controllers
drifting apart rather than two seeds being independent.

So the protocol status of both certificates is reported per pair (the PROTO
column, the ``protocol`` line of the detail block, a warning naming the file,
and a column in both report formats), and ``--require-scenario-hashes`` -- the
flag that says "I want this audit to rest on scenario digests" -- fails any pair
that is not ``on`` on both sides. Five ways a pair is not:

    off        both certificates carry the off-protocol marker
    mixed      one does; the pair is not even internally comparable
    unstamped  a certificate carries no ``scenario_protocol`` key at all, so
               whether its episodes were keyed is unknown. Every current writer
               stamps the field (scripts/test_certificate_scenario_stamp.py),
               so this means a certificate written before the protocol existed
               or by something that is not one of those writers.
    invalid    a certificate carries the key but its value is not a usable
               header: ``null``, a bare string, an array, an empty object, an
               object with no ``version``, or a ``version`` that is not a
               positive integer. PRESENCE IS NOT COMPLIANCE -- the status is
               read from the SHAPE and CONTENT of the header, never from the
               key merely existing.
    stale      a well-formed header declaring a protocol version other than
               the current one (``SCENARIO_PROTOCOL_VERSION`` in
               tasks/_shared/scenario_rng.py, read from that source by AST so
               a bump cannot leave this gate behind). Version 1 is the
               historical global-torch-RNG behaviour whose elimination is the
               reason the protocol exists, so a certificate declaring it must
               never pass the gate.

Both carry the failing certificate's name and the exact defect into the PROTO
column, the summary counts, the warnings, both report formats and the
``--require-scenario-hashes`` fail reason.

Without the flag these stay warnings: an off-protocol pair is still worth
auditing (a CONFIRMED_DUPLICATE verdict on one is still a duplicate), it just
must not be read as a clean paired comparison.

Caveat on --float-tol: q is always computed from exact values, so a non-zero
tolerance makes q (and therefore ``chance``) a slight under-estimate of how
easily the channel could match. Keep the default of 0.0 unless you are hunting
for a replay blurred by GPU non-determinism.

Verdicts:
    CONFIRMED_DUPLICATE     >=1 IDENTICAL channel and no DIFFERING channel.
                            The two seeds ran the same scenarios; the pair is
                            one measurement reported twice.
    PARTIAL_SHORT_CIRCUIT   an ANOMALOUS channel, or IDENTICAL and DIFFERING
                            channels at the same time. Some random primitives
                            follow the evaluation seed and some do not, so the
                            two seeds are correlated but not equal -- invisible
                            to a success-pattern census.
    CONFIRMED_INDEPENDENT   >=1 DIFFERING channel and no IDENTICAL/ANOMALOUS
                            one. Every scenario channel this certificate
                            EXPOSES moved with the seed. Note the coverage
                            column: without scenario hashes an old certificate
                            only exposes spawn/goal geometry (or nothing but
                            outcomes), so this is "no evidence of duplication",
                            not proof that every primitive was reseeded.
    DEGENERATE_UNDECIDABLE  nothing informative either way -- classically an
                            all-succeed or all-fail pair with no discriminating
                            continuous field. Binary comparison proves nothing.
                            NOT a pass: the evidence is missing, not clean.

Exit codes: 0 clean, 1 at least one CONFIRMED_DUPLICATE or PARTIAL_SHORT_CIRCUIT
(or a --fail-on-degenerate / --require-scenario-hashes violation, the latter
including an off-protocol, mixed, unstamped, invalid or stale pair), 2 bad usage
(including: the current protocol version cannot be read from its source, which
fails closed rather than guessing). Every failing pair is listed with the reason
it failed, so a gate log says which rule fired.

    python scripts/check_scenario_independence.py gcerts_cache gcerts gcerts_pid
    python scripts/check_scenario_independence.py "gcerts*/*.json" --report a.md
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
import math
import os
import re
import sys
from collections import Counter

# ---------------------------------------------------------------------------
# Per-episode field taxonomy. Names are the ones eval_v6_frozen.py actually
# writes (see the ``rec = {...}`` block and the hasattr-guarded additions).
# ---------------------------------------------------------------------------

# Latched from the layout at reset, so a pure function of the SCENARIO: the
# controller cannot move them. d0_m/route_geodesic_m are written straight from
# the spawn/goal geometry (tasks/hazard_nav/hazard_nav_env.py:1775-1777).
SCENARIO_FIELDS = ("d0_m", "route_geodesic_m")

# Depend on scenario AND controller.
OUTCOME_FIELDS = (
    "tts_s",
    "path_length_m",
    "min_clearance_m",
    "contact_steps",
    "contact_seconds",
    "contact_longest_seconds",
    "contact_depth_mean_m",
    "gates",
    "max_phase",
)

BINARY_FIELD = "success"

# Episode identity inside a certificate.
KEY_FIELDS = ("env", "ep")

# If these disagree the two files are not the same experiment at two seeds and
# must not be paired at all (same guard as check_eval_seed_independence.py).
IDENTITY_FIELDS = ("task", "level", "checkpoint", "controller",
                   "train_task", "eval_task")

# Metadata that legitimately drifts between two runs of the SAME arm because it
# is a counter over the run, not a configuration. Reported, never blocking.
VOLATILE_META_KEYS = ("straight_fallbacks", "legs_planned", "git_commit")

# Not per-episode measurements.
NON_CHANNEL_FIELDS = set(KEY_FIELDS) | {
    "controller", "scenario_hash", "scenario_hashes",
}

LAYER_HASH = "hash"
LAYER_SCENARIO = "scenario"
LAYER_OUTCOME = "outcome"
LAYER_BINARY = "binary"

VERDICT_ORDER = (
    "CONFIRMED_DUPLICATE",
    "PARTIAL_SHORT_CIRCUIT",
    "DEGENERATE_UNDECIDABLE",
    "CONFIRMED_INDEPENDENT",
)
FAILING_VERDICTS = ("CONFIRMED_DUPLICATE", "PARTIAL_SHORT_CIRCUIT")

# The value an evaluator stamps into "scenario_protocol" when the env it ran
# carried no ScenarioRNG. Duplicated rather than imported: this file is
# deliberately stdlib-only so it can run as a CI gate on a machine with no
# torch and no Isaac Sim, while tasks/_shared/scenario_draws.py:349 (the
# definition) and scripts/eval_v6_frozen.py:91 (the other copy) both live
# behind a torch import. The three are pinned equal by
# scripts/test_check_scenario_independence.py, which reads the literal out of
# scenario_draws.py by text.
SCENARIO_PROTOCOL_OFF = "off-protocol"

# Per-pair protocol status. Only "on" means the (env, ep) keys this auditor
# aligns on name the same scenario on both sides.
PROTOCOL_ON = "on"
PROTOCOL_OFF = "off"
PROTOCOL_MIXED = "mixed"
PROTOCOL_UNSTAMPED = "unstamped"
PROTOCOL_INVALID = "invalid"
PROTOCOL_STALE = "stale"

# Every status the report can print, in the order the summary counts them.
PROTOCOL_STATUSES = (PROTOCOL_ON, PROTOCOL_OFF, PROTOCOL_MIXED,
                     PROTOCOL_UNSTAMPED, PROTOCOL_INVALID, PROTOCOL_STALE)

# The key inside the header block that carries the protocol version, and the
# name of the constant that defines the CURRENT one.
PROTOCOL_VERSION_KEY = "version"
PROTOCOL_VERSION_CONSTANT = "SCENARIO_PROTOCOL_VERSION"

# Source of truth for the current protocol version. Read by AST rather than
# imported for the same reason the off-protocol literal is duplicated above:
# tasks/_shared/scenario_rng.py imports numpy and torch at module scope, and
# this auditor has to run as a CI gate on a machine with neither. Reading the
# constant instead of pinning a literal here means a protocol bump is picked
# up by the gate on the same commit that makes it, with no second edit that
# could be forgotten -- and forgetting it would leave the gate accepting
# certificates from the retired protocol.
SCENARIO_RNG_SOURCE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tasks", "_shared", "scenario_rng.py")


class ProtocolVersionUnavailable(RuntimeError):
    """The current protocol version could not be read from the source.

    Fail closed and loud. The alternative -- carrying on with a guessed or
    defaulted version -- is what this whole change exists to stop: a gate that
    cannot tell which protocol is current cannot tell a compliant header from
    a retired one, and would wave both through.
    """


_PROTOCOL_VERSION_CACHE = {}


def read_protocol_version(path):
    """Parse ``SCENARIO_PROTOCOL_VERSION`` out of ``path`` without importing.

    Mirrors the AST-reading precedent in
    scripts/test_check_scenario_independence.py (which pins the off-protocol
    literal against tasks/_shared/scenario_draws.py). Both plain and annotated
    module-level assignments are accepted; anything else -- no assignment, two
    assignments, a non-integer -- raises rather than guessing.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
    except (OSError, UnicodeDecodeError) as error:
        # UnicodeDecodeError is a ValueError, not an OSError, so it has to be
        # named: an undecodable source is still an unreadable one.
        raise ProtocolVersionUnavailable(
            "cannot read {} ({})".format(path, error)) from error
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as error:
        raise ProtocolVersionUnavailable(
            "cannot parse {} ({})".format(path, error)) from error

    found = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if not isinstance(node.value, ast.Constant):
            continue
        if any(isinstance(target, ast.Name)
               and target.id == PROTOCOL_VERSION_CONSTANT
               for target in targets):
            found.append(node.value.value)

    if len(found) != 1 or isinstance(found[0], bool) \
            or not isinstance(found[0], int):
        raise ProtocolVersionUnavailable(
            "{} does not define {} as exactly one integer constant "
            "(found {!r})".format(path, PROTOCOL_VERSION_CONSTANT, found))
    return found[0]


def current_protocol_version(source=None):
    """The protocol version a certificate must declare to count as "on".

    Cached per source path so a run over hundreds of certificates parses the
    file once, and so a test can point the auditor at a different source file
    and get a fresh read.
    """
    path = source or SCENARIO_RNG_SOURCE
    if path not in _PROTOCOL_VERSION_CACHE:
        _PROTOCOL_VERSION_CACHE[path] = read_protocol_version(path)
    return _PROTOCOL_VERSION_CACHE[path]


def json_type_name(value):
    """The JSON type name of a decoded value, for a reason string."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def short_json(value, limit=60):
    """A compact, ASCII, single-line rendering of a decoded JSON value."""
    try:
        text = json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        text = repr(value)
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[:limit - 3] + "..."
    return text


def sanitise_reason(text):
    """No ';' inside one reason: the CSV fail_reasons column joins with ';'."""
    return text.replace(";", ",")


def classify_protocol_field(stamped, value, version):
    """Classify one certificate's ``scenario_protocol`` field.

    Returns ``(status, problem)`` where ``problem`` is ``None`` for the three
    states that are not a defect (on, off, unstamped) and a reason string
    naming exactly what is wrong otherwise.

    PRESENCE IS NOT COMPLIANCE. Before this, the auditor read any value that
    was not the off-protocol literal as "on", so ``null``, an arbitrary
    string, ``{}`` and -- worst -- an explicit ``{"version": 1}`` header all
    classified as on protocol and PASSED --require-scenario-hashes. Version 1
    is the historical global-torch-RNG behaviour whose elimination is the
    entire point of the protocol, so a certificate declaring it is the one
    thing the gate must never accept.
    """
    if not stamped:
        return PROTOCOL_UNSTAMPED, None
    if value == SCENARIO_PROTOCOL_OFF:
        return PROTOCOL_OFF, None
    if not isinstance(value, dict):
        return PROTOCOL_INVALID, (
            "scenario_protocol is {} ({}), not the header block and not the "
            "{!r} marker".format(json_type_name(value), short_json(value),
                                 SCENARIO_PROTOCOL_OFF))
    if PROTOCOL_VERSION_KEY not in value:
        return PROTOCOL_INVALID, (
            "scenario_protocol header carries no {!r} key (keys: {})".format(
                PROTOCOL_VERSION_KEY,
                ", ".join(sorted(str(k) for k in value)) or "none"))
    declared = value[PROTOCOL_VERSION_KEY]
    if isinstance(declared, bool) or not isinstance(declared, int):
        return PROTOCOL_INVALID, (
            "scenario_protocol version is {} ({}), not an integer".format(
                json_type_name(declared), short_json(declared)))
    if declared < 1:
        return PROTOCOL_INVALID, (
            "scenario_protocol version {} is not a positive integer".format(
                declared))
    if declared != version:
        note = (" -- version 1 is the historical global-torch-RNG behaviour "
                "the protocol replaced" if declared == 1 else "")
        return PROTOCOL_STALE, (
            "scenario_protocol declares version {} but the current protocol "
            "is version {}{}".format(declared, version, note))
    return PROTOCOL_ON, None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class Certificate:
    """One certificate JSON, normalised."""

    def __init__(self, path, payload):
        self.path = path
        self.stem = os.path.splitext(os.path.basename(path))[0]
        self.dirname = os.path.dirname(os.path.abspath(path))
        self.payload = payload
        self.seed = payload.get("seed")
        self.records = payload.get("records") or []
        self.meta = {k: v for k, v in payload.items() if k != "records"}
        self.notes = []

        self.by_key = {}
        for index, record in enumerate(self.records):
            key = (record.get("env"), record.get("ep"))
            if key in self.by_key:
                self.notes.append(
                    "duplicate episode key {}; keeping the first".format(key))
                continue
            self.by_key[key] = (index, record)

        self.arm = strip_eval_seed_token(self.stem, self.seed)
        stem_seed = filename_seed(self.stem)
        if stem_seed is not None and self.seed is not None \
                and stem_seed != self.seed:
            self.notes.append(
                "filename says eval seed {} but the JSON says {}".format(
                    stem_seed, self.seed))

        self.protocol = payload.get("scenario_protocol")
        # A positive declaration that this run's scenarios were NOT keyed by
        # (protocol version, eval seed, env, episode, group). Compared against
        # the literal only: the header block is a dict, so no truthiness test
        # can confuse the two, and an absent key is a THIRD state (unstamped),
        # not a quiet pass.
        self.off_protocol = self.protocol == SCENARIO_PROTOCOL_OFF
        self.protocol_unstamped = "scenario_protocol" not in payload
        # SHAPE AND CONTENT, not mere presence: a header that is not a mapping,
        # or carries no integer version, or declares a version other than the
        # current one gets its own status and its own reason string instead of
        # being waved through as "on".
        self.protocol_state, self.protocol_problem = classify_protocol_field(
            not self.protocol_unstamped, self.protocol,
            current_protocol_version())
        self.hashes = extract_scenario_hashes(payload, self.records,
                                              self.by_key)

    @property
    def episode_count(self):
        return len(self.by_key)


def filename_seed(stem):
    match = re.search(r"[_\-]e(\d+)$", stem)
    return int(match.group(1)) if match else None


def strip_eval_seed_token(stem, seed):
    """Drop the evaluation-seed token so the two seeds share one arm key.

    Only ``e``-prefixed tokens are stripped. ``cert_ringseal_s42_e123`` keeps
    its ``s42`` (that is the TRAINING seed and it distinguishes two different
    arms) and loses only ``_e123``.
    """
    if seed is not None:
        for token in ("_e{}".format(seed), "-e{}".format(seed),
                      "_eval{}".format(seed), "_evalseed{}".format(seed),
                      ".e{}".format(seed)):
            position = stem.rfind(token)
            if position >= 0:
                return stem[:position] + stem[position + len(token):]
    return re.sub(r"[_\-]e\d+$", "", stem)


def extract_scenario_hashes(payload, records, by_key):
    """Return {primitive: {(env, ep): hexdigest}}, empty when absent.

    Precedence: per-record ``scenario_hashes`` > per-record ``scenario_hash`` >
    top-level ``scenario_hashes``. See the module docstring for the schema.
    """
    out = {}

    def put(primitive, key, value):
        out.setdefault(str(primitive), {})[key] = str(value)

    saw_per_record = False
    for record in records:
        key = (record.get("env"), record.get("ep"))
        blob = record.get("scenario_hashes")
        if blob is None:
            blob = record.get("scenario_hash")
        if blob is None:
            continue
        saw_per_record = True
        if isinstance(blob, dict):
            for primitive, value in blob.items():
                put(primitive, key, value)
        else:
            put("scenario", key, blob)
    if saw_per_record:
        return out

    top = payload.get("scenario_hashes")
    if isinstance(top, dict):
        index_to_key = {index: key for key, (index, _) in by_key.items()}
        for primitive, values in top.items():
            if isinstance(values, dict):
                for raw_key, value in values.items():
                    key = parse_episode_key(raw_key)
                    if key is not None:
                        put(primitive, key, value)
            elif isinstance(values, list):
                for index, value in enumerate(values):
                    key = index_to_key.get(index)
                    if key is not None:
                        put(primitive, key, value)
    return out


def parse_episode_key(raw):
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return (raw[0], raw[1])
    text = str(raw)
    for separator in (":", ",", "|", "/"):
        if separator in text:
            left, _, right = text.partition(separator)
            try:
                return (int(left), int(right))
            except ValueError:
                return None
    return None


def expand_inputs(patterns, recursive):
    """Directories, globs and plain paths -> a sorted list of .json files."""
    found = []
    for pattern in patterns:
        if os.path.isdir(pattern):
            if recursive:
                found.extend(glob.glob(os.path.join(pattern, "**", "*.json"),
                                       recursive=True))
            else:
                found.extend(glob.glob(os.path.join(pattern, "*.json")))
        elif os.path.isfile(pattern):
            found.append(pattern)
        else:
            found.extend(glob.glob(pattern, recursive=recursive))
    seen, ordered = set(), []
    for path in sorted(found):
        real = os.path.abspath(path)
        if real not in seen:
            seen.add(real)
            ordered.append(path)
    return ordered


def load_certificates(paths):
    certificates, skipped = [], []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError) as error:
            skipped.append((path, "unreadable: {}".format(error)))
            continue
        if not isinstance(payload, dict):
            skipped.append((path, "top level is {}, not a certificate object"
                            .format(type(payload).__name__)))
            continue
        records = payload.get("records")
        if not isinstance(records, list) or not records:
            skipped.append((path, "no 'records' list"))
            continue
        if not isinstance(records[0], dict):
            skipped.append((path, "'records' does not hold episode objects"))
            continue
        certificates.append(Certificate(path, payload))
    return certificates, skipped


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

def strip_volatile(value):
    """Copy of a metadata value with run counters removed, for comparison."""
    if isinstance(value, dict):
        return {k: strip_volatile(v) for k, v in value.items()
                if k not in VOLATILE_META_KEYS}
    if isinstance(value, list):
        return [strip_volatile(v) for v in value]
    return value


def build_pairs(certificates):
    """Group by (directory, arm) and pair up differing evaluation seeds."""
    groups = {}
    for certificate in certificates:
        groups.setdefault((certificate.dirname, certificate.arm), []) \
            .append(certificate)

    pairs, unpaired, blocked = [], [], []
    for (_dirname, arm), members in sorted(groups.items(),
                                           key=lambda kv: (kv[0][0], kv[0][1])):
        members = sorted(members,
                         key=lambda c: (c.seed is None, c.seed, c.stem))
        if len(members) < 2:
            unpaired.append((members[0],
                             "no sibling certificate at another evaluation "
                             "seed"))
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if a.seed == b.seed:
                    blocked.append((a, b, "both files record evaluation seed "
                                          "{}".format(a.seed)))
                    continue
                mismatched = [field for field in IDENTITY_FIELDS
                              if a.meta.get(field) != b.meta.get(field)]
                if mismatched:
                    blocked.append((a, b, "not the same experiment; differs on "
                                          + ", ".join(mismatched)))
                    continue
                pairs.append((arm, a, b))
    return pairs, unpaired, blocked


def soft_metadata_diff(a, b):
    keys = set(a.meta) | set(b.meta)
    keys.discard("seed")
    diffs = []
    for key in sorted(keys):
        if strip_volatile(a.meta.get(key)) != strip_volatile(b.meta.get(key)):
            diffs.append(key)
    return diffs


# ---------------------------------------------------------------------------
# Channel comparison
# ---------------------------------------------------------------------------

class Channel:
    def __init__(self, name, layer, n_cmp, n_match, q, status):
        self.name = name
        self.layer = layer
        self.n_cmp = n_cmp
        self.n_match = n_match
        self.q = q
        self.status = status

    @property
    def frac(self):
        return self.n_match / self.n_cmp if self.n_cmp else 0.0

    @property
    def chance(self):
        if self.layer == LAYER_HASH:
            return 0.0
        if not self.n_cmp:
            return 1.0
        return self.q ** self.n_cmp

    def describe(self):
        return ("{name:<24s} {layer:<8s} match {m:>4d}/{n:<4d} "
                "frac={f:6.3f} q={q:5.3f} chance={c:9.3g}  {s}").format(
            name=self.name, layer=self.layer, m=self.n_match, n=self.n_cmp,
            f=self.frac, q=self.q, c=self.chance, s=self.status)


def values_equal(left, right, tol):
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) == bool(right)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, float) and math.isnan(left):
            return isinstance(right, float) and math.isnan(right)
        if isinstance(right, float) and math.isnan(right):
            return False
        if tol <= 0.0:
            return left == right
        return abs(float(left) - float(right)) <= tol
    return left == right


def _hashable(value):
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value


def collision_probability(values):
    """P(two independent episodes of this run carry the same value)."""
    usable = [v for v in values if v is not None]
    if not usable:
        return 1.0
    counts = Counter(_hashable(v) for v in usable)
    n = len(usable)
    return sum((c / n) ** 2 for c in counts.values())


def compare_channel(name, layer, left_values, right_values, tol,
                    chance_threshold):
    indices = [i for i in range(len(left_values))
               if not (left_values[i] is None and right_values[i] is None)]
    if not indices:
        return Channel(name, layer, 0, 0, 1.0, "SKIPPED")
    n = len(indices)
    matches = sum(1 for i in indices
                  if values_equal(left_values[i], right_values[i], tol))
    q = max(collision_probability([left_values[i] for i in indices]),
            collision_probability([right_values[i] for i in indices]))
    channel = Channel(name, layer, n, matches, q, "DIFFERING")

    if layer == LAYER_HASH:
        if matches == n:
            channel.status = "IDENTICAL"
        elif matches > 0:
            channel.status = "ANOMALOUS"
        return channel

    if matches == n:
        channel.status = ("IDENTICAL" if channel.chance <= chance_threshold
                          else "UNINFORMATIVE_MATCH")
        return channel

    noise = 3.0 * math.sqrt(max(q * (1.0 - q), 0.0) / n) + 1.0 / n
    if q < 1.0 and channel.frac > min(q + noise, 1.0):
        channel.status = "ANOMALOUS"
    return channel


def channels_for_pair(a, b, keys, tol, chance_threshold):
    channels = []

    for primitive in sorted(set(a.hashes) | set(b.hashes)):
        left = a.hashes.get(primitive, {})
        right = b.hashes.get(primitive, {})
        # Only episodes hashed on BOTH sides can be compared. A primitive that
        # one certificate stamps and the other does not is a schema gap, not
        # evidence of a different scenario -- comparing a hash against a
        # missing value would manufacture a DIFFERING channel and read as
        # false independence.
        shared = [k for k in keys if k in left and k in right]
        if not shared:
            channels.append(
                Channel("hash:" + primitive, LAYER_HASH, 0, 0, 1.0, "SKIPPED"))
            continue
        channels.append(compare_channel(
            "hash:" + primitive, LAYER_HASH,
            [left[k] for k in shared], [right[k] for k in shared],
            0.0, chance_threshold))

    ordered_fields = [BINARY_FIELD]
    ordered_fields.extend(SCENARIO_FIELDS)
    ordered_fields.extend(OUTCOME_FIELDS)
    # Anything else per-episode and scalar that the schema grows later.
    extra = set()
    for cert in (a, b):
        for key in keys:
            for field, value in cert.by_key[key][1].items():
                if field in NON_CHANNEL_FIELDS or field in ordered_fields:
                    continue
                if value is None or isinstance(value, (int, float, bool, str)):
                    extra.add(field)
    ordered_fields.extend(sorted(extra))

    for field in ordered_fields:
        present = any(field in a.by_key[k][1] or field in b.by_key[k][1]
                      for k in keys)
        if not present:
            continue
        layer = (LAYER_BINARY if field == BINARY_FIELD else
                 LAYER_SCENARIO if field in SCENARIO_FIELDS else LAYER_OUTCOME)
        channels.append(compare_channel(
            field, layer,
            [a.by_key[k][1].get(field) for k in keys],
            [b.by_key[k][1].get(field) for k in keys],
            tol, chance_threshold))
    return channels


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

class PairResult:
    def __init__(self, arm, a, b):
        self.arm = arm
        self.a = a
        self.b = b
        self.display_arm = arm
        self.keys = []
        self.channels = []
        self.verdict = "DEGENERATE_UNDECIDABLE"
        self.reason = ""
        self.warnings = []
        # Filled by main(); a pair fails iff this is non-empty, and each entry
        # names the rule that fired so a gate log is self-explaining.
        self.fail_reasons = []

    @property
    def n_common(self):
        return len(self.keys)

    def protocol_status(self):
        """One of the PROTOCOL_STATUSES values, for this pair.

        Anything but "on" means the two certificates cannot be assumed to have
        drawn the same scenario for the same (env, ep) key, so every channel
        verdict below rests on a premise the certificates do not support.

        Most severe wins. A rejected header ("invalid") and a retired protocol
        version ("stale") outrank the marker states: a certificate whose stamp
        is junk, or which declares the protocol whose defect this audit exists
        to find, says LESS about its episode keying than one that honestly
        stamps itself off protocol.
        """
        states = [c.protocol_state for c in (self.a, self.b)]
        if PROTOCOL_INVALID in states:
            return PROTOCOL_INVALID
        if PROTOCOL_STALE in states:
            return PROTOCOL_STALE
        off = states.count(PROTOCOL_OFF)
        if off == 2:
            return PROTOCOL_OFF
        if off:
            return PROTOCOL_MIXED
        if PROTOCOL_UNSTAMPED in states:
            return PROTOCOL_UNSTAMPED
        return PROTOCOL_ON

    def off_protocol_sides(self):
        """The certificates of this pair carrying the off-protocol marker."""
        return [c for c in (self.a, self.b) if c.off_protocol]

    def unstamped_sides(self):
        """The certificates of this pair with no scenario_protocol key."""
        return [c for c in (self.a, self.b) if c.protocol_unstamped]

    def rejected_protocol_sides(self):
        """(certificate, reason) for each side whose header failed validation.

        Empty for on / off / unstamped: those three are states of a WELL-FORMED
        field, not defects in it.
        """
        return [(c, c.protocol_problem) for c in (self.a, self.b)
                if c.protocol_problem]

    def protocol_problem_text(self):
        """The rejected sides as one line, for a warning or a fail reason."""
        return " / ".join(
            "{}: {}".format(os.path.basename(certificate.path), why)
            for certificate, why in self.rejected_protocol_sides())

    def by_status(self, status):
        return [c for c in self.channels if c.status == status]

    def success_channel(self):
        for channel in self.channels:
            if channel.name == BINARY_FIELD:
                return channel
        return None

    def hash_channels(self):
        return [c for c in self.channels
                if c.layer == LAYER_HASH and c.status != "SKIPPED"]

    def skipped_hash_channels(self):
        return [c for c in self.channels
                if c.layer == LAYER_HASH and c.status == "SKIPPED"]

    def continuous_channels(self):
        return [c for c in self.channels
                if c.layer in (LAYER_SCENARIO, LAYER_OUTCOME)]

    def coverage(self):
        if self.hash_channels():
            return "per-primitive hashes ({})".format(len(self.hash_channels()))
        if any(c.layer == LAYER_SCENARIO and c.status != "SKIPPED"
               for c in self.channels):
            return "spawn/goal geometry only"
        if any(c.layer == LAYER_OUTCOME and c.status != "SKIPPED"
               for c in self.channels):
            return "outcomes only"
        return "none"


def classify(result):
    if not result.keys:
        result.verdict = "DEGENERATE_UNDECIDABLE"
        result.reason = ("the two certificates share no (env, ep) episode key, "
                         "so nothing can be compared")
        return result

    identical = result.by_status("IDENTICAL")
    anomalous = result.by_status("ANOMALOUS")
    differing = result.by_status("DIFFERING")

    if anomalous:
        result.verdict = "PARTIAL_SHORT_CIRCUIT"
        result.reason = ("{} agrees on {}/{} episodes -- more than chance "
                         "allows but not everywhere, so some primitives follow "
                         "the eval seed and some do not".format(
                             anomalous[0].name, anomalous[0].n_match,
                             anomalous[0].n_cmp))
    elif identical and differing:
        result.verdict = "PARTIAL_SHORT_CIRCUIT"
        result.reason = ("{} is pinned across the two seeds while {} moves -- "
                         "the scenario is only partly reseeded".format(
                             ", ".join(c.name for c in identical[:3]),
                             ", ".join(c.name for c in differing[:3])))
    elif identical:
        result.verdict = "CONFIRMED_DUPLICATE"
        result.reason = ("{} identical on every compared episode and nothing "
                         "differs: the two seeds ran the same scenarios".format(
                             ", ".join(c.name for c in identical[:4])))
    elif differing:
        result.verdict = "CONFIRMED_INDEPENDENT"
        result.reason = ("every channel this certificate exposes moves with the "
                         "seed (coverage: {})".format(result.coverage()))
    else:
        result.verdict = "DEGENERATE_UNDECIDABLE"
        uninformative = result.by_status("UNINFORMATIVE_MATCH")
        if uninformative:
            result.reason = ("everything matches but every match is free ({}): "
                             "an all-succeed / all-fail or constant-valued "
                             "channel proves nothing".format(
                                 ", ".join(c.name for c in uninformative[:4])))
        else:
            result.reason = "no channel carries usable information"
    return result


def disambiguate_arms(results):
    """Prefix the arm with its directory when the same arm name lives in two.

    gcert_pid_gateT5 exists in both gcerts_cache/ and gcerts_pid/; two rows
    with the identical label would be unreadable in a gate log.
    """
    directories = {}
    for result in results:
        directories.setdefault(result.arm, set()).add(result.a.dirname)
    for result in results:
        if len(directories[result.arm]) > 1:
            result.display_arm = "{}/{}".format(
                os.path.basename(result.a.dirname), result.arm)


def audit_pair(arm, a, b, tol, chance_threshold, require_hashes):
    result = PairResult(arm, a, b)
    result.keys = sorted(set(a.by_key) & set(b.by_key),
                         key=lambda k: (k[0] is None, k))
    result.channels = channels_for_pair(a, b, result.keys, tol,
                                        chance_threshold)
    classify(result)

    for certificate in (a, b):
        for note in certificate.notes:
            result.warnings.append("{}: {}".format(
                os.path.basename(certificate.path), note))
    dropped_a = a.episode_count - result.n_common
    dropped_b = b.episode_count - result.n_common
    if dropped_a or dropped_b:
        result.warnings.append(
            "episode keys not shared: {} only in seed {}, {} only in seed {}"
            .format(dropped_a, a.seed, dropped_b, b.seed))
    diffs = soft_metadata_diff(a, b)
    if diffs:
        result.warnings.append("metadata also differs on: " + ", ".join(diffs))
    if a.protocol != b.protocol:
        result.warnings.append("scenario_protocol differs: {} vs {}".format(
            a.protocol, b.protocol))
    # The marker, said out loud. Before this, two off-protocol certificates
    # audited together produced a report in which the string "off-protocol"
    # never appeared and the run exited 0.
    off_sides = result.off_protocol_sides()
    if off_sides:
        result.warnings.append(
            "OFF PROTOCOL: {} stamped scenario_protocol={!r}, so those "
            "episode scenarios are not keyed by (eval seed, env, episode) and "
            "the (env, ep) keys compared above need not name the same scenario "
            "on both sides -- this pair is not a scenario-paired comparison in "
            "either direction".format(
                ", ".join(os.path.basename(c.path) for c in off_sides),
                SCENARIO_PROTOCOL_OFF))
    # A header that does not validate is worse than a missing one: it LOOKS
    # like a compliance claim. Name the file and the exact defect.
    rejected = result.rejected_protocol_sides()
    if rejected:
        result.warnings.append(
            "PROTOCOL HEADER REJECTED ({}): {} -- a scenario_protocol header "
            "counts as compliance only when it is a mapping declaring the "
            "current integer protocol version ({}); this pair is NOT a "
            "protocol-paired comparison".format(
                result.protocol_status(), result.protocol_problem_text(),
                current_protocol_version()))
    unstamped = result.unstamped_sides()
    if unstamped:
        result.warnings.append(
            "no scenario_protocol field in {} -- written before the protocol "
            "existed, or by a writer that does not stamp it; whether its "
            "episodes are keyed is UNKNOWN, not confirmed".format(
                ", ".join(os.path.basename(c.path) for c in unstamped)))
    gaps = result.skipped_hash_channels()
    if gaps:
        result.warnings.append(
            "scenario hashes present on one side only for: "
            + ", ".join(c.name.split(":", 1)[-1] for c in gaps)
            + " -- not comparable, so those primitives are UNAUDITED")
    if require_hashes and not result.hash_channels():
        result.warnings.append(
            "REQUIRED: no scenario hashes in this pair "
            "(--require-scenario-hashes); the verdict rests on indirect "
            "numeric evidence")
    if require_hashes and result.protocol_status() != PROTOCOL_ON:
        result.warnings.append(
            "REQUIRED: pair protocol status is {!r} "
            "(--require-scenario-hashes); scenario hashes can only be paired "
            "episode by episode when BOTH sides ran on the protocol".format(
                result.protocol_status()))
    return result


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def short_task(task):
    if not isinstance(task, str):
        return str(task)
    text = re.sub(r"^Isaac-USV-", "", task)
    return re.sub(r"-Direct-v\d+$", "", text)


def channel_summary(channels):
    if not channels:
        return "-"
    identical = sum(1 for c in channels if c.status == "IDENTICAL")
    anomalous = sum(1 for c in channels if c.status == "ANOMALOUS")
    usable = sum(1 for c in channels if c.status != "SKIPPED")
    text = "{}/{} id".format(identical, usable)
    if anomalous:
        text += " +{}anom".format(anomalous)
    return text


def success_cell(result):
    channel = result.success_channel()
    if channel is None or channel.status == "SKIPPED":
        return "n/a"
    if channel.n_match == channel.n_cmp:
        return "yes*" if channel.status == "UNINFORMATIVE_MATCH" else "yes"
    return "no"


def hash_cell(result):
    channels = result.hash_channels()
    if not channels:
        return "-"
    identical = [c for c in channels if c.status == "IDENTICAL"]
    return "{}/{} id".format(len(identical), len(channels))


def describe_protocol(certificate):
    """One certificate's stamp, short enough for a detail line.

    Re-derives the per-side status from the raw value rather than reading an
    attribute, so the rendering cannot disagree with the classification.
    """
    if certificate.off_protocol:
        return SCENARIO_PROTOCOL_OFF
    if certificate.protocol_unstamped:
        return "unstamped"
    protocol = certificate.protocol
    state, _problem = classify_protocol_field(True, protocol,
                                              current_protocol_version())
    if isinstance(protocol, dict):
        if PROTOCOL_VERSION_KEY in protocol:
            text = "v{}/{}".format(protocol.get(PROTOCOL_VERSION_KEY),
                                   protocol.get("hash"))
        else:
            text = "no-version"
    else:
        text = short_json(protocol, 24)
    if state == PROTOCOL_STALE:
        return text + " (stale)"
    if state == PROTOCOL_INVALID:
        return text + " (invalid)"
    return text


def protocol_cell(result):
    """PROTO column: "on" is the only value the audit's premise holds under."""
    status = result.protocol_status()
    return status if status == PROTOCOL_ON else status.upper()


def render_table(results, stream):
    header = ("{:<32s} {:<22s} {:>9s} {:>5s} {:<5s} {:<14s} {:<8s} {:<9s} {}"
              .format("ARM", "TASK", "SEEDS", "N", "SUCC", "CONTINUOUS",
                      "HASHES", "PROTO", "VERDICT"))
    stream.write(header + "\n")
    stream.write("-" * len(header) + "\n")
    for result in results:
        stream.write(
            "{:<32s} {:<22s} {:>9s} {:>5d} {:<5s} {:<14s} {:<8s} {:<9s} {}\n"
            .format(
                result.display_arm[:32],
                short_task(result.a.meta.get("task"))[:22],
                "{}/{}".format(result.a.seed, result.b.seed),
                result.n_common,
                success_cell(result),
                channel_summary(result.continuous_channels()),
                hash_cell(result),
                protocol_cell(result),
                result.verdict))


def render_details(result, stream, show_all_channels):
    stream.write("\n{} [{}]\n".format(result.arm, result.verdict))
    stream.write("  files    : {}\n             {}\n".format(
        result.a.path, result.b.path))
    stream.write("  task     : {} level={} seeds {} vs {}\n".format(
        result.a.meta.get("task"), result.a.meta.get("level"),
        result.a.seed, result.b.seed))
    stream.write("  episodes : {} compared (of {} and {})\n".format(
        result.n_common, result.a.episode_count, result.b.episode_count))
    stream.write("  coverage : {}\n".format(result.coverage()))
    stream.write("  protocol : {} ({} | {})\n".format(
        result.protocol_status(),
        describe_protocol(result.a), describe_protocol(result.b)))
    stream.write("  reason   : {}\n".format(result.reason))
    for channel in result.channels:
        if not show_all_channels and channel.status == "SKIPPED":
            continue
        stream.write("    " + channel.describe() + "\n")
    for warning in result.warnings:
        stream.write("  ! {}\n".format(warning))


def write_report(path, results, unpaired, blocked, skipped):
    if os.path.splitext(path)[1].lower() in (".md", ".markdown"):
        _write_markdown(path, results, unpaired, blocked, skipped)
    else:
        _write_csv(path, results)


def _write_markdown(path, results, unpaired, blocked, skipped):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Scenario independence audit\n\n")
        handle.write("| arm | task | seeds | episodes | success identical | "
                     "continuous | hashes | protocol | coverage | verdict | "
                     "reason |\n")
        handle.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for result in results:
            handle.write("| {} | {} | {}/{} | {} | {} | {} | {} | {} | {} | "
                         "**{}** | {} |\n".format(
                             result.display_arm,
                             short_task(result.a.meta.get("task")),
                             result.a.seed, result.b.seed, result.n_common,
                             success_cell(result),
                             channel_summary(result.continuous_channels()),
                             hash_cell(result), protocol_cell(result),
                             result.coverage(),
                             result.verdict,
                             result.reason.replace("|", "/")))
        handle.write("\n## Per-channel detail\n")
        for result in results:
            handle.write("\n### {} ({})\n\n```\n".format(
                result.arm, result.verdict))
            for channel in result.channels:
                if channel.status == "SKIPPED":
                    continue
                handle.write(channel.describe() + "\n")
            for warning in result.warnings:
                handle.write("! " + warning + "\n")
            handle.write("```\n")
        handle.write("\n## Not compared\n\n")
        for certificate, why in unpaired:
            handle.write("- unpaired `{}`: {}\n".format(certificate.path, why))
        for a, b, why in blocked:
            handle.write("- blocked `{}` vs `{}`: {}\n".format(
                a.path, b.path, why))
        for path_, why in skipped:
            handle.write("- skipped `{}`: {}\n".format(path_, why))


def _write_csv(path, results):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "arm", "task", "level", "seed_a", "seed_b", "file_a", "file_b",
            "episodes_compared", "success_identical", "continuous_identical",
            "continuous_compared", "anomalous_channels", "hash_primitives",
            "hash_primitives_identical", "protocol_status", "protocol_a",
            "protocol_b", "coverage", "verdict", "fail_reasons", "reason",
            "identical_channels", "differing_channels", "warnings",
        ])
        for result in results:
            continuous = result.continuous_channels()
            hashes = result.hash_channels()
            writer.writerow([
                result.display_arm,
                result.a.meta.get("task"),
                result.a.meta.get("level"),
                result.a.seed,
                result.b.seed,
                result.a.path,
                result.b.path,
                result.n_common,
                success_cell(result),
                sum(1 for c in continuous if c.status == "IDENTICAL"),
                sum(1 for c in continuous if c.status != "SKIPPED"),
                ";".join(c.name for c in result.by_status("ANOMALOUS")),
                ";".join(c.name.split(":", 1)[-1] for c in hashes),
                ";".join(c.name.split(":", 1)[-1] for c in hashes
                         if c.status == "IDENTICAL"),
                result.protocol_status(),
                describe_protocol(result.a),
                describe_protocol(result.b),
                result.coverage(),
                result.verdict,
                ";".join(result.fail_reasons),
                result.reason,
                ";".join(c.name for c in result.by_status("IDENTICAL")),
                ";".join(c.name for c in result.by_status("DIFFERING")),
                " | ".join(result.warnings),
            ])


LEGEND = """
VERDICT RULES
  CONFIRMED_DUPLICATE    at least one channel is identical on every compared
                         episode with a chance-of-luck below the threshold, and
                         no channel differs. The two evaluation seeds ran the
                         SAME scenarios: their agreement is arithmetic, not
                         reproducibility evidence.
  PARTIAL_SHORT_CIRCUIT  a channel agrees more than chance allows but not
                         everywhere, or one channel is pinned while another
                         moves. Some random primitives are seed-derived and
                         some still come from a clobbered global RNG, so the
                         two seeds are correlated. A success-pattern census
                         cannot see this.
  CONFIRMED_INDEPENDENT  something differs and nothing is pinned. Scope is only
                         as wide as the coverage line: certificates without
                         scenario hashes expose spawn/goal geometry at best.
  DEGENERATE_UNDECIDABLE every match is free (all-succeed / all-fail success,
                         constant-valued fields) and nothing differs, so the
                         comparison proves neither way. Missing evidence, not a
                         pass.
  SUCC column            'yes' identical success pattern, 'yes*' identical but
                         degenerate (all-succeed / all-fail: no information),
                         'no' patterns differ.
  PROTO column           'on' both certificates declare a WELL-FORMED header
                         for the CURRENT protocol version, so an (env, ep) key
                         names the SAME scenario on both sides and every
                         verdict above means what it says.
                         'OFF' both carry the off-protocol marker, 'MIXED' one
                         does, 'UNSTAMPED' a certificate does not declare the
                         field at all, 'INVALID' a header is not a mapping or
                         carries no integer version (null, a bare string, {}),
                         'STALE' a header declares a protocol version that is
                         not the current one -- version 1 being exactly the
                         global-RNG behaviour the protocol replaced. In those
                         five the episodes are not known to be paired, so a
                         DIFFERING channel may be two controllers drifting
                         rather than two independent seeds;
                         --require-scenario-hashes fails them, naming the
                         certificate and the defect.
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Audit certificate pairs for evaluation-seed scenario "
                    "independence.")
    parser.add_argument("inputs", nargs="+",
                        help="certificate directories, globs or files")
    parser.add_argument("--report", default=None,
                        help="write a report; .md/.markdown -> markdown, "
                             "anything else -> CSV")
    parser.add_argument("--chance-threshold", type=float, default=1e-2,
                        help="a total match counts as evidence only if the "
                             "probability of it happening by luck is below "
                             "this (default 0.01)")
    parser.add_argument("--float-tol", type=float, default=0.0,
                        help="absolute tolerance for numeric equality "
                             "(default 0.0 = bit-exact, which is what an "
                             "identical scenario under a deterministic policy "
                             "produces)")
    parser.add_argument("--recursive", action="store_true",
                        help="descend into sub-directories of input dirs")
    parser.add_argument("--verbose", action="store_true",
                        help="print per-channel detail for every pair, not "
                             "only the failing ones")
    parser.add_argument("--require-scenario-hashes", action="store_true",
                        help="fail pairs that carry no per-primitive scenario "
                             "hashes, and pairs whose protocol status is not "
                             "'on' (off-protocol, mixed, unstamped, invalid or "
                             "stale): hashes are only comparable episode by "
                             "episode when both certificates declare a "
                             "well-formed header for the current protocol "
                             "version")
    parser.add_argument("--fail-on-degenerate", action="store_true",
                        help="also fail on DEGENERATE_UNDECIDABLE pairs")
    parser.add_argument("--quiet", action="store_true",
                        help="table and summary only")
    return parser


def main(argv=None, stream=None):
    args = build_parser().parse_args(argv)
    stream = stream or sys.stdout

    # Fail closed BEFORE reading a single certificate: an auditor that cannot
    # tell which protocol version is current cannot tell a compliant header
    # from a retired one, and must not pretend to.
    try:
        version = current_protocol_version()
    except ProtocolVersionUnavailable as error:
        stream.write("cannot determine the current scenario protocol "
                     "version: {}\n".format(error))
        return 2

    paths = expand_inputs(args.inputs, args.recursive)
    if not paths:
        stream.write("no JSON files matched: {}\n".format(
            " ".join(args.inputs)))
        return 2

    certificates, skipped = load_certificates(paths)
    pairs, unpaired, blocked = build_pairs(certificates)

    results = [audit_pair(arm, a, b, args.float_tol, args.chance_threshold,
                          args.require_scenario_hashes)
               for arm, a, b in pairs]
    disambiguate_arms(results)
    results.sort(key=lambda r: (VERDICT_ORDER.index(r.verdict),
                                r.display_arm))

    # One pass, one place: each rule that can fail a pair records WHY on the
    # pair, so a pair failing for two reasons is listed once with both, and the
    # gate log names the rule instead of only the verdict. Computed BEFORE
    # anything is rendered: the CSV carries a fail_reasons column, and filling
    # it after the report was written left that column empty in every report
    # this auditor has ever produced.
    for result in results:
        if result.verdict in FAILING_VERDICTS:
            result.fail_reasons.append(result.verdict)
        if args.fail_on_degenerate \
                and result.verdict == "DEGENERATE_UNDECIDABLE":
            result.fail_reasons.append(
                "DEGENERATE_UNDECIDABLE (--fail-on-degenerate)")
        if args.require_scenario_hashes:
            status = result.protocol_status()
            if status != PROTOCOL_ON:
                # The audit cannot do its job here: without the protocol on
                # both sides the (env, ep) alignment is not known to compare
                # the same scenario, so neither a match nor a mismatch is
                # evidence about the evaluation seeds. When the field itself
                # was rejected, say what was wrong with it -- "INVALID" alone
                # sends the reader back to the certificate to guess.
                detail = result.protocol_problem_text()
                result.fail_reasons.append(sanitise_reason(
                    "{}_PROTOCOL (--require-scenario-hashes){}".format(
                        status.upper(),
                        ": " + detail if detail else "")))
            if not result.hash_channels():
                result.fail_reasons.append(
                    "NO_SCENARIO_HASHES (--require-scenario-hashes)")
    failed = [r for r in results if r.fail_reasons]

    stream.write("=== SCENARIO INDEPENDENCE AUDIT ===\n")
    stream.write("inputs               : {}\n".format(" ".join(args.inputs)))
    stream.write("certificates loaded  : {} ({} files skipped)\n".format(
        len(certificates), len(skipped)))
    stream.write("seed pairs formed    : {}\n".format(len(results)))
    stream.write("chance threshold     : {:g}   float tolerance: {:g}\n"
                 .format(args.chance_threshold, args.float_tol))
    stream.write("scenario protocol    : version {} required "
                 "(read from {})\n\n".format(
                     version, os.path.basename(SCENARIO_RNG_SOURCE)))

    if results:
        render_table(results, stream)

    if not args.quiet:
        for result in results:
            if args.verbose or result.verdict in FAILING_VERDICTS \
                    or result.verdict == "DEGENERATE_UNDECIDABLE":
                render_details(result, stream, args.verbose)

    counts = Counter(result.verdict for result in results)
    stream.write("\nSUMMARY\n")
    for verdict in VERDICT_ORDER:
        stream.write("  {:<24s} {}\n".format(verdict, counts.get(verdict, 0)))
    # Printed even under --quiet, and even when nothing failed: a pair whose
    # certificates were never on the protocol is not a paired comparison, and a
    # summary that says only "PASS" hides that.
    protocol_counts = Counter(result.protocol_status() for result in results)
    stream.write("  {:<24s} {}\n".format(
        "scenario protocol",
        ", ".join("{} {}".format(protocol_counts.get(status, 0), status)
                  for status in PROTOCOL_STATUSES)))
    # The reason, not only the count: "1 invalid" alone does not say which
    # certificate or what was wrong with it.
    for result in results:
        if result.protocol_status() in (PROTOCOL_INVALID, PROTOCOL_STALE):
            stream.write("  {:<24s} {}: {}\n".format(
                "  rejected header", result.display_arm,
                result.protocol_problem_text()))

    if unpaired or blocked or skipped:
        stream.write("\nNOT COMPARED\n")
        for certificate, why in unpaired:
            stream.write("  unpaired {}: {}\n".format(certificate.path, why))
        for a, b, why in blocked:
            stream.write("  blocked  {} vs {}: {}\n".format(
                os.path.basename(a.path), os.path.basename(b.path), why))
        for path, why in skipped:
            stream.write("  skipped  {}: {}\n".format(path, why))

    if not args.quiet:
        stream.write(LEGEND)

    if args.report:
        write_report(args.report, results, unpaired, blocked, skipped)
        stream.write("\nreport written to {}\n".format(args.report))

    if failed:
        stream.write("\nFAIL: {} pair(s) are not an independent-seed "
                     "reproducibility check:\n".format(len(failed)))
        for result in failed:
            # The verdict already occupies its own column, so only the rules
            # that are NOT the verdict itself are worth repeating here.
            extra = [why for why in result.fail_reasons
                     if why != result.verdict]
            stream.write("  {:<24s} {:<32s} {}\n".format(
                result.verdict, result.display_arm[:32],
                ", ".join(extra) if extra else "-"))
        return 1
    stream.write("\nPASS: no duplicated or partially short-circuited seed "
                 "pair.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
